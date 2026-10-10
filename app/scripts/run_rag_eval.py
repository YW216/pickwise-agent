"""RAG 检索评测：Hit@K / MRR / Recall@K / Precision@K 三方基线对照。

数据：app/evaluation/rag_cases.json（review_status=approved，防幻觉流程产出）；
负例无 ground truth，不进指标（单独计数）。

判定粒度（两层，呼应指标含义）：
- 知识类：**section 级**（doc+section 全匹配）定 Hit@K / MRR——衡量排序精度；
  **doc 级**定 Recall@K / Precision@K——单标注下 Recall 退化为"期望文档是否召回"，
  Precision@3 = top3 中来自期望文档的比例（上下文噪声代理）
- 商品类：product_id 定 Hit@K / MRR / Precision@K（单商品标注，Precision=hit/K，解释受限）

三方对照：
  dense          —— MilvusBackend.search（纯 dense 对照组）
  hybrid-narrow  —— hybrid，recall_k = top_k（旧行为：召回窗口=输出量）
  hybrid-wide    —— hybrid，recall_k = 20（当前生产默认：漏斗召回）

评测与生产共用检索路径（知识走 MilvusBackend，商品走 search_products 的
_hybrid_product_ids），保证测的就是线上行为。

用法：
  python app/scripts/run_rag_eval.py                # 三方全跑，top_k=5（生产口径）
  python app/scripts/run_rag_eval.py --top-k 3     # 严格口径
  python app/scripts/run_rag_eval.py --limit 10     # 冒烟：只跑前 10 条 approved

产出：终端对照表 + app/evaluation/runs/rag_report.json（三模式指标与未命中清单，
供 run-to-run 对比；--no-save 可关闭）；未命中清单也逐模式打印，便于定位语料缺口。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RUNS_DIR = ROOT / "app" / "evaluation" / "runs"

from pymilvus import MilvusClient  # noqa: E402

from app.agent.rag.embedder import Embedder  # noqa: E402
from app.agent.rag.milvus_backend import MilvusBackend  # noqa: E402
from app.agent.rag.milvus_utils import ensure_reachable  # noqa: E402
from app.db.snapshot import PRODUCTS  # noqa: E402
from app.agent.tools.search_products import _hybrid_product_ids  # noqa: E402
from app.config.settings import settings  # noqa: E402

# 默认口径对齐生产（search_products 默认 limit=5）；严格口径跑 --top-k 3
TOP_K = 5


def _resolve_uri() -> str:
    return settings.milvus_uri


def _hit_mrr(case, results, kind) -> tuple[bool, float]:
    """section 级（知识）/ product_id 级（商品）命中与 MRR。"""
    if kind == "knowledge":
        ranks = [
            i + 1
            for i, h in enumerate(results)
            if h.chunk.doc == case["expected_doc"]
            and h.chunk.section == case["expected_section"]
        ]
    else:
        ranks = [
            i + 1
            for i, pid in enumerate(results)
            if pid == case["expected_product_id"]
        ]
    return bool(ranks), (1.0 / ranks[0] if ranks else 0.0)


def _doc_metrics(case, results) -> tuple[bool, float]:
    """doc 级 Recall / Precision（仅知识类：上下文噪声代理）。"""
    doc_hits = sum(1 for h in results if h.chunk.doc == case["expected_doc"])
    return bool(doc_hits), doc_hits / len(results) if results else 0.0


def main():
    parser = argparse.ArgumentParser(description="RAG 检索评测（三方基线对照）")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条 approved（冒烟）")
    parser.add_argument("--wide-recall", type=int, default=20, help="hybrid-wide 的召回窗口")
    parser.add_argument(
        "--cases", default="app/evaluation/rag_cases.json", help="评测集路径"
    )
    parser.add_argument(
        "--include-pending", action="store_true",
        help="把 pending 状态的 case 也纳入评测（新生成未抽检时的基线口径）",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="不落盘 runs/rag_report.json（默认落盘，供 run-to-run 对比）",
    )
    args = parser.parse_args()

    uri = _resolve_uri()
    ensure_reachable(uri)

    cases_path = ROOT / args.cases
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    runnable = [
        c for c in data["cases"]
        if c["review_status"] == "approved"
        or (args.include_pending and c["review_status"] == "pending")
    ]
    negatives = [c for c in data["cases"] if c["type"] == "negative"]
    if args.limit:
        runnable = runnable[: args.limit]
    knowledge_cases = [c for c in runnable if c["type"] == "knowledge"]
    product_cases = [c for c in runnable if c["type"] == "product"]
    print(
        f"评测集：纳入 {len(runnable)}"
        f"（知识 {len(knowledge_cases)} / 商品 {len(product_cases)}），"
        f"负例 {len(negatives)} 条不进指标 | top_k={args.top_k}"
        + ("｜含 pending（未抽检基线口径）" if args.include_pending else "")
    )

    # ---------- 检索器（与生产同路径） ----------
    settings.assert_embedding_configured()
    embedder = Embedder(
        api_key=settings.effective_embedding_api_key,
        base_url=settings.effective_embedding_base_url,
        model=settings.effective_embedding_model,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )
    kb_backend = MilvusBackend(uri=uri, collection_name=settings.milvus_collection)
    kb_backend.load()

    product_client = MilvusClient(uri)
    if not product_client.has_collection(settings.product_collection):
        print("❌ product_kb 不存在，请先运行 build_product_kb.py")
        sys.exit(1)
    product_client.load_collection(settings.product_collection)

    # ---------- 批量编码（一次 embedding 调用，三模式复用） ----------
    k_vectors = embedder.encode([c["query"] for c in knowledge_cases])
    p_vectors = embedder.encode([c["query"] for c in product_cases])

    MODES = ["dense", "hybrid-narrow", "hybrid-wide"]

    def _retrieve_knowledge(mode, text, vec):
        if mode == "dense":
            return kb_backend.search(vec, top_k=args.top_k)
        rk = args.top_k if mode == "hybrid-narrow" else args.wide_recall
        return kb_backend.hybrid_search(text, vec, top_k=args.top_k, recall_k=rk)

    def _retrieve_product(mode, text, vec, case):
        # 对齐生产：Agent 从 query 识别价格意图后传 max_price 标量过滤（expr 下推两路）
        mp = case.get("max_price")
        expr = f"price <= {int(mp)}" if mp else None
        if mode == "dense":
            hits = product_client.search(
                collection_name=settings.product_collection,
                data=[vec],
                anns_field="embedding",
                search_params={"metric_type": "COSINE"},
                filter=expr or "",
                limit=args.top_k,
                output_fields=["product_id"],
            )[0]
            return [h["entity"].get("product_id", h.get("id", "")) for h in hits]
        rk = args.top_k if mode == "hybrid-narrow" else args.wide_recall
        return _hybrid_product_ids(text, vec, top_k=args.top_k, recall_k=rk, expr=expr)

    # ---------- 三方 × 两类逐 case 评测 ----------
    stats: dict[str, dict[str, list]] = {}
    misses: dict[str, list[str]] = {}
    for mode in MODES:
        stats[mode] = {
            "knowledge": [], "product": [],
            "k_doc_recall": [], "k_doc_precision": [],
        }
        misses[mode] = []

    print("\n检索中（每 mode 约 65 次查询）...")
    for i, case in enumerate(knowledge_cases):
        vec = k_vectors[i]
        for mode in MODES:
            results = _retrieve_knowledge(mode, case["query"], vec)
            hit, mrr = _hit_mrr(case, results, "knowledge")
            doc_recall, doc_precision = _doc_metrics(case, results)
            stats[mode]["knowledge"].append((hit, mrr))
            stats[mode]["k_doc_recall"].append(doc_recall)
            stats[mode]["k_doc_precision"].append(doc_precision)
            if not hit:
                misses[mode].append(case["case_id"])
        if (i + 1) % 10 == 0:
            print(f"   知识进度 {i + 1}/{len(knowledge_cases)}")

    for i, case in enumerate(product_cases):
        vec = p_vectors[i]
        # 多标注：期望 = 可接受商品集合（expected_product_ids，兼容旧单值字段）
        expected = set(
            case.get("expected_product_ids") or [case["expected_product_id"]]
        )
        for mode in MODES:
            ids = _retrieve_product(mode, case["query"], vec, case)
            ranks = [i + 1 for i, pid in enumerate(ids) if pid in expected]
            hit = bool(ranks)
            mrr = 1.0 / ranks[0] if ranks else 0.0
            precision = (
                sum(1 for pid in ids if pid in expected) / len(ids) if ids else 0.0
            )
            stats[mode]["product"].append((hit, mrr, precision))
            if not hit:
                misses[mode].append(case["case_id"])
        if (i + 1) % 10 == 0:
            print(f"   商品进度 {i + 1}/{len(product_cases)}")

    # ---------- 汇总输出 ----------
    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("\n" + "=" * 74)
    k_label = f"Hit@{args.top_k}"
    print(f"{'模式':<16}{'类别':<10}{'n':>4}{k_label:>9}{'MRR':>9}{'Recall@'+str(args.top_k):>10}{'Prec@'+str(args.top_k):>9}")
    print("-" * 74)
    for mode in MODES:
        s = stats[mode]
        k_hits = [h for h, _ in s["knowledge"]]
        k_mrr = [m for _, m in s["knowledge"]]
        p_rows = s["product"]
        p_hits = [h for h, _, _ in p_rows]
        p_mrr = [m for _, m, _ in p_rows]
        p_prec = [pr for _, _, pr in p_rows]
        print(
            f"{mode:<16}{'knowledge':<10}{len(k_hits):>4}"
            f"{_mean(k_hits):>9.3f}{_mean(k_mrr):>9.3f}"
            f"{_mean(s['k_doc_recall']):>10.3f}{_mean(s['k_doc_precision']):>9.3f}"
        )
        print(
            f"{mode:<16}{'product':<10}{len(p_hits):>4}"
            f"{_mean(p_hits):>9.3f}{_mean(p_mrr):>9.3f}"
            f"{_mean(p_hits):>10.3f}{_mean(p_prec):>9.3f}"
        )
    print("-" * 74)
    print("知识类：Hit/MRR 为 section 级（排序精度），Recall/Precision 为 doc 级（噪声代理）；")
    print("商品类：多标注（可接受商品集合），Hit/MRR/Prec 均对集合判定。")
    for mode in MODES:
        if misses[mode]:
            print(f"[{mode}] 未命中: {', '.join(misses[mode])}")
    print(f"\n基线完成。验收流程固定化：改动 → 重跑本脚本 → 对照本表。")

    # ---------- 结果落盘（批次 3）：供 run-to-run 对比，别再只活在终端里 ----------
    if not args.no_save:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        summary = {
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "top_k": args.top_k,
            "wide_recall": args.wide_recall,
            "included": {
                "knowledge": len(knowledge_cases),
                "product": len(product_cases),
                "negatives": len(negatives),
                "include_pending": args.include_pending,
            },
            "modes": {
                mode: {
                    "knowledge": {
                        "n": len(stats[mode]["knowledge"]),
                        "hit_at_k": _mean([h for h, _ in stats[mode]["knowledge"]]),
                        "mrr": _mean([m for _, m in stats[mode]["knowledge"]]),
                        "doc_recall": _mean(stats[mode]["k_doc_recall"]),
                        "doc_precision": _mean(stats[mode]["k_doc_precision"]),
                    },
                    "product": {
                        "n": len(stats[mode]["product"]),
                        "hit_at_k": _mean([h for h, _, _ in stats[mode]["product"]]),
                        "mrr": _mean([m for _, m, _ in stats[mode]["product"]]),
                        "precision": _mean([pr for _, _, pr in stats[mode]["product"]]),
                    },
                    "misses": misses[mode],
                }
                for mode in MODES
            },
        }
        # 归一化文件名的模式键（hybrid-narrow → hybrid_narrow），便于 jq 取值
        for mode in MODES:
            out = RUNS_DIR / f"rag_{mode.replace('-', '_')}.json"
            out.write_text(
                json.dumps({**summary, "mode": mode}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        (RUNS_DIR / "rag_report.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果已落盘: {RUNS_DIR / 'rag_report.json'}（含三模式指标与未命中清单）")


if __name__ == "__main__":
    main()
