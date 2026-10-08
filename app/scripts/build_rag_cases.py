"""构建 RAG 检索评测集（防幻觉流程：种子 chunk → LLM 生成 query → 程序校验）。

防幻觉三道防线（核心原则：AI 的输出只进候选，不进事实）：
1. 生成方向反转：种子 = 语料中的真实 chunk，LLM 只负责"把这段内容改写成
   用户口吻的 query"——期望结果就是种子本身（引用即原文），标注零幻觉空间
2. 程序校验：去重 / 长度 / 期望 section 与商品的存在性——凡可代码验证的
   绝不靠 AI，幻觉在这一步被结构性杀死
3. LLM 只做过滤器：负例（编造型号）生成后校验"型号确实不存在于商品库"，
   校验失败直接丢弃——AI 判错的代价是丢一条评测（保守），不是注入错误标注

用法：
  python app/scripts/build_rag_cases.py                  # 全量：49 知识 + 16 商品 + 5 负例
  python app/scripts/build_rag_cases.py --knowledge-limit 10 --negatives 3

产物：
  app/evaluation/rag_cases.json        评测集（人工抽检后作为 run_rag_eval.py 输入）
  app/evaluation/rag_cases_review.md   人工抽检清单（case_id + query + 期望结果）
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from app.agent.rag.chunker import chunk_markdown_dir  # noqa: E402
from app.db.snapshot import PRODUCTS  # noqa: E402
from app.config.settings import settings  # noqa: E402

# 卡片渲染与 build_product_kb 共用同一模板，保证评测 query 与索引内容同源
from app.scripts.build_product_kb import build_card_text  # noqa: E402

# query 风格轮换：避免 65 条 query 全是同一种问法（分布多样性）
STYLES = [
    ("口语随口问", "像用户随口一问，口语化，不要书面腔"),
    ("具体需求", "带明确的使用场景或需求描述"),
    ("对比/咨询", "带有选择、比较、咨询的语气"),
    ("参数关注", "关注某个具体参数或配置"),
    ("场景描述", "描述一个使用场景，让商品/知识被动匹配"),
]

SYSTEM_PROMPT = "你是评测集构造助手。只输出一个 JSON 对象，不要输出任何其他内容。"


def _llm_query(client: OpenAI, model: str, user_prompt: str, attempts: int = 3) -> dict | None:
    """调 LLM 生成 query，返回解析后的 dict。

    内置指数退避重试：批跑时 deepseek 会限流（429）/超时，异常吞成 None 会让
    种子被误跳过——这里对异常重试 3 次（间隔 1.5s/3s），JSON 解析失败同样重试。
    """
    import re
    import time

    last_err = None
    for attempt in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                    max_tokens=4096,
                    # 思考模型：多标注 prompt（挑 33 候选）reasoning 约 1700 token，
                    # 1024 预算被思考耗尽 → content 空（finish=length）。4096 才能覆盖思考+正文
                    reasoning_effort="low",
                    timeout=30,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").removeprefix("json").strip()
            m = re.search(r"\{.*\}", raw, re.S)  # 容错：正文前后混入说明文字时提取 JSON 块
            if m:
                raw = m.group(0)
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception as exc:  # noqa: BLE001 —— 429/超时/解析失败统一退避重试
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"   [warn] LLM 调用 {attempts} 次失败: {type(last_err).__name__}: {last_err}")
    return None


def _valid_query(query) -> bool:
    # 上限 100：LLM 倾向把"一句话"写成连环三问（实测 50~70 字常见），80 会误杀
    return isinstance(query, str) and 4 <= len(query.strip()) <= 100


def _norm(query: str) -> str:
    return "".join(query.split()).lower()


def main():
    parser = argparse.ArgumentParser(description="构建 RAG 检索评测集（防幻觉流程）")
    parser.add_argument("--knowledge-limit", type=int, default=0, help="知识 chunk 抽样数，0=全量")
    parser.add_argument("--product-limit", type=int, default=0, help="商品抽样数，0=全量")
    parser.add_argument("--negatives", type=int, default=5, help="负例（编造型号）条数")
    parser.add_argument("--out", default="app/evaluation/rag_cases.json")
    parser.add_argument("--no-resume", action="store_true", help="忽略已有评测集，从头生成")
    parser.add_argument(
        "--backfill-price", action="store_true",
        help="价格标注补全：为商品 case 从 query 抽取 max_price（对齐生产 Agent 的标量过滤行为），跑完即退出",
    )
    parser.add_argument(
        "--complete-labels", action="store_true",
        help="标注补全：检索 top5 交 LLM 复核可接受商品集合，只扩不缩（修标注缺口型假阴性），跑完即退出",
    )
    args = parser.parse_args()

    client = OpenAI(
        api_key=settings.openai_api_key, base_url=settings.openai_base_url,
        timeout=settings.openai_timeout, max_retries=settings.openai_max_retries,
    )
    model = settings.model_name
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 价格标注补全模式（失败归因 ①：价格约束必须走标量过滤） ----------
    if args.backfill_price:
        store = json.loads(out_path.read_text(encoding="utf-8"))
        pcases = [
            c for c in store["cases"]
            if c["type"] == "product" and "max_price" not in c
        ]
        print(f"价格标注补全：{len(pcases)} 条商品 case 待抽取 max_price")
        filled = 0
        for i, case in enumerate(pcases):
            prompt = (
                f"以下是一条电商购物需求：\n「{case['query']}」\n\n"
                f"请判断其中是否包含价格约束，并给出检索可用的价格上限 max_price（整数，元）：\n"
                f"- 「X元内 / X以下」→ X\n"
                f"- 「X多」（如两千多）→ 该档整数上界（两千多 → 2999）\n"
                f"- 「X左右 / X上下」→ X*1.1 取整到十位\n"
                f"- 「A到B元」→ B\n"
                f"- 若无价格约束 → null\n\n"
                f'输出 JSON：{{"max_price": 整数或 null}}'
            )
            data = _llm_query(client, model, prompt)
            mp = data.get("max_price") if data else None
            if isinstance(mp, int) and 100 <= mp <= 20000:
                case["max_price"] = mp
                filled += 1
            else:
                case["max_price"] = None
            if (i + 1) % 20 == 0:
                print(f"   进度 {i + 1}/{len(pcases)}，有价格约束 {filled} 条")
            time.sleep(0.6)
        store["meta"]["price_annotated"] = {
            "total": len(pcases), "with_max_price": filled,
            "note": "对齐生产：Agent 从 query 识别价格意图后传 max_price 标量过滤",
        }
        out_path.write_text(
            json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"🎉 价格标注完成：{filled}/{len(pcases)} 条含 max_price → {out_path}")
        return

    # ---------- 标注补全模式（失败归因 ②：标注缺口制造假阴性） ----------
    if args.complete_labels:
        from app.agent.tools.search_products import _get_client, _hybrid_product_ids
        from app.agent.rag.embedder import Embedder
        from app.agent.rag.milvus_utils import ensure_reachable

        store = json.loads(out_path.read_text(encoding="utf-8"))
        pcases = [c for c in store["cases"] if c["type"] == "product"]
        uri = settings.milvus_uri
        uri = uri if uri.startswith("http") else str(ROOT / uri)
        ensure_reachable(uri)
        product_client = _get_client()
        product_client.load_collection(settings.product_collection)
        embedder = Embedder(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            timeout=settings.openai_timeout,
            max_retries=settings.openai_max_retries,
        )
        vecs = embedder.encode([c["query"] for c in pcases])
        print(f"标注补全：复核 {len(pcases)} 条商品 case 的 top5（对齐生产：top_k=5 + max_price 过滤）")

        expanded = 0
        for i, (case, vec) in enumerate(zip(pcases, vecs)):
            mp = case.get("max_price")
            expr = f"price <= {int(mp)}" if mp else None
            ids = _hybrid_product_ids(case["query"], vec, top_k=5, expr=expr)
            expected = set(case.get("expected_product_ids") or [case["expected_product_id"]])
            cards = "\n\n".join(
                f"[{pid}] {PRODUCTS[pid]['name']}（{PRODUCTS[pid]['price']} 元）\n"
                + build_card_text(PRODUCTS[pid])[:220]
                for pid in ids if pid in PRODUCTS
            )
            prompt = (
                f"用户需求：\n「{case['query']}」\n\n"
                f"以下是检索到的商品：\n{cards}\n\n"
                f"已标注的可接受商品：{sorted(expected)}\n\n"
                f"请判断：检索结果里哪些商品也**明显满足**这条需求的全部核心要求？\n"
                f"规则：只能从上面列出的商品 id 中选；只选明显满足的，宁缺毋滥；"
                f"已标注的商品不用重复输出；没有则输出空数组。\n"
                f'输出 JSON：{{"acceptable": ["id", ...]}}'
            )
            data = _llm_query(client, model, prompt)
            acceptable = [
                pid for pid in (data.get("acceptable") or [])
                if isinstance(pid, str) and pid in ids and pid not in expected
            ] if data else []
            if acceptable:
                case["expected_product_ids"] = sorted(expected | set(acceptable))
                case["review_note"] = (
                    (case.get("review_note", "") + f" | 标注补全 +{acceptable}").strip(" |")
                )
                expanded += 1
            if (i + 1) % 20 == 0:
                print(f"   进度 {i + 1}/{len(pcases)}，扩集合 {expanded} 条")
            time.sleep(0.6)
        store["meta"]["labels_completed"] = {
            "total": len(pcases), "expanded": expanded,
            "note": "标注补全：miss 的 top5 由 LLM 复核，只扩不缩；扩集合记录在 review_note 供审计",
        }
        out_path.write_text(
            json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"🎉 标注补全完成：{expanded}/{len(pcases)} 条扩了可接受集合 → {out_path}")
        return

    # ---------- 种子抽取（程序从语料抽样，AI 不参与"哪里来"） ----------
    # 概览块治理：目录式低密度块不进索引（见 build_kb_index 同款过滤），也不做种子
    knowledge_seeds = []
    for c in chunk_markdown_dir(ROOT / settings.kb_dir):
        if c.chunk_id.endswith("#00-00"):
            continue
        knowledge_seeds.append({
            "source_chunk_id": c.chunk_id,
            "expected_doc": c.doc,
            "expected_section": c.section,
            "content": c.text,
        })
    product_seeds = []
    for pid, p in PRODUCTS.items():
        # 同品类候选（多标注的可挑集合）：LLM 只能从真实卡片池挑，不得编造
        peers = "\n".join(
            f"- {q['product_id']} {q['name']}（{q['price']} 元）"
            for q in PRODUCTS.values()
            if q["category"] == p["category"] and q["product_id"] != pid
        )
        product_seeds.append({
            "source_chunk_id": pid,
            "expected_product_id": pid,
            "name": p["name"],
            "category": p["category"],
            "content": build_card_text(p),
            "peers": peers,
        })
    if args.knowledge_limit:
        knowledge_seeds = knowledge_seeds[: args.knowledge_limit]
    if args.product_limit:
        product_seeds = product_seeds[: args.product_limit]
    print(f"种子：知识 {len(knowledge_seeds)} + 商品 {len(product_seeds)} + 负例 {args.negatives}")

    # ---------- 断点续跑：复用已有 case（不重复花 LLM 调用），只补缺失种子 ----------
    cases: list[dict] = []
    seen: set[str] = set()
    covered: set[str] = set()
    used_ids: set[str] = set()
    if not args.no_resume and out_path.exists():
        old = json.loads(out_path.read_text(encoding="utf-8"))
        cases = old.get("cases", [])
        for c in cases:
            seen.add(_norm(c["query"]))
            used_ids.add(c["case_id"])
            if c.get("source_chunk_id"):
                covered.add(c["source_chunk_id"])
        print(f"续跑：已加载 {len(cases)} 条历史 case（覆盖 {len(covered)} 个种子）")

    skipped = 0

    def _unique_id(base: str) -> str:
        """case_id 去重（续跑时新 case 不得与历史 id 冲突）。"""
        id_, n = base, 1
        while id_ in used_ids:
            n += 1
            id_ = f"{base}-{n}"
        used_ids.add(id_)
        return id_

    def _try_add(case: dict, query) -> bool:
        """程序校验 + 去重；通过返回 True。"""
        nonlocal skipped
        if not _valid_query(query):
            skipped += 1
            return False
        norm = _norm(query)
        if norm in seen:
            skipped += 1
            return False
        seen.add(norm)
        case["query"] = query.strip()
        cases.append(case)
        return True

    # ---------- 知识 chunk → query（期望 = 种子本身，零幻觉） ----------
    print("\n[1/3] 知识 chunk 生成 query...")
    for i, seed in enumerate(knowledge_seeds):
        if seed["source_chunk_id"] in covered:
            continue
        style, style_desc = STYLES[i % len(STYLES)]
        prompt = (
            f"以下是电商知识库中的一段内容：\n【{seed['expected_doc']} · {seed['expected_section']}】\n"
            f"{seed['content']}\n\n"
            f"请站在用户角度，写一条{style_desc}的自然语言查询，这段内容能直接回答它。\n"
            f"要求：只依据上述内容，不要引入其中不存在的型号/参数/政策承诺；15~40 个字。\n"
            f'输出 JSON：{{"query": "..."}}'
        )
        case = {
            "case_id": _unique_id(f"KC-{i + 1:03d}"),
            "type": "knowledge",
            "style": style,
            "expected_doc": seed["expected_doc"],
            "expected_section": seed["expected_section"],
            "source_chunk_id": seed["source_chunk_id"],
            "review_status": "pending",
        }
        data = _llm_query(client, model, prompt)
        if data is None or not _try_add(case, data.get("query")):
            data = _llm_query(client, model, prompt)  # 解析/校验失败重试一次
            if data is None or not _try_add(case, data.get("query")):
                skipped += 1  # 统计修正：两轮都失败才是真跳过
                print(f"   跳过 {seed['source_chunk_id']}（生成/校验失败）")
        if (i + 1) % 10 == 0:
            print(f"   进度 {i + 1}/{len(knowledge_seeds)}，累计 {len(cases)} 条")
        time.sleep(0.8)  # 调用间隔，避免触发限流

    # ---------- 商品卡片 → query（多标注：期望 = 可接受商品集合） ----------
    print(f"\n[2/3] 商品卡片生成 query（多标注）...")
    for i, seed in enumerate(product_seeds):
        if seed["source_chunk_id"] in covered:
            continue
        style, style_desc = STYLES[i % len(STYLES)]
        prompt = (
            f"以下是商品卡片（种子）：\n{seed['content']}\n\n"
            f"同品类其他商品（挑选候选，只能从中选，不得编造）：\n{seed['peers']}\n\n"
            f"请站在买家角度，写一条会让**种子商品**被检索到的自然语言需求（{style_desc}），"
            f"然后从种子+候选中挑出**所有**能满足这条需求的商品 id"
            f"（通常 1~3 个；明显不满足的不要挑，宁缺毋滥，种子必须包含）。\n"
            f"要求：query 只依据卡片信息，不要编造不存在的参数或卖点；15~40 个字。\n"
            f'输出 JSON：{{"query": "...", "accepted": ["{seed["expected_product_id"]}", ...]}}'
        )
        case = {
            "case_id": _unique_id(f"PC-{i + 1:03d}"),
            "type": "product",
            "style": style,
            "expected_product_id": seed["expected_product_id"],
            "expected_product_ids": [seed["expected_product_id"]],
            "source_chunk_id": seed["source_chunk_id"],
            "review_status": "pending",
        }
        data = _llm_query(client, model, prompt)
        if data is None or not _try_add(case, data.get("query")):
            data = _llm_query(client, model, prompt)
            if data is None or not _try_add(case, data.get("query")):
                skipped += 1  # 统计修正
                print(f"   跳过 {seed['source_chunk_id']}（生成/校验失败）")
                time.sleep(0.8)
                continue
        # 多标注校验：accepted 全部真实存在、必须包含种子、去重——LLM 只能挑不能编
        accepted_raw = data.get("accepted") or []
        accepted_ids = sorted({
            pid for pid in accepted_raw
            if isinstance(pid, str) and pid in PRODUCTS
        })
        if seed["expected_product_id"] not in accepted_ids:
            accepted_ids.insert(0, seed["expected_product_id"])
        case["expected_product_ids"] = accepted_ids
        if len(accepted_ids) > 1:
            case["review_note"] = f"多标注：可接受集合 {accepted_ids}"
        time.sleep(0.8)  # 调用间隔，避免触发限流

    # ---------- 负例：编造型号（程序校验"确实不存在"） ----------
    print(f"\n[3/3] 负例生成（编造不存在的型号）...")
    neg_existing = sum(1 for c in cases if c["type"] == "negative")
    neg_target = max(0, args.negatives - neg_existing)
    brands = sorted({p["brand"] for p in PRODUCTS.values()})
    neg_made = 0
    neg_attempts = 0
    while neg_made < neg_target and neg_attempts < neg_target * 3 + 3:
        neg_attempts += 1
        prompt = (
            f"现有商品库品牌：{'、'.join(brands)}。\n"
            f"请编造一个听起来合理、但下列商品库中**不存在**的商品型号查询，"
            f"风格像真实买家提问（品牌可沿用上述品牌，但型号必须是虚构的新名字）。\n"
            f'输出 JSON：{{"query": "...", "model": "虚构的完整型号名"}}'
        )
        data = _llm_query(client, model, prompt)
        if not data or not _valid_query(data.get("query")) or not data.get("model"):
            continue
        invented = str(data["model"])
        # 程序校验"确实不存在"：虚构型号不得是任何现有商品名/型号的子串（反向亦然）
        collision = any(
            invented in p["name"] or p["name"] in invented for p in PRODUCTS.values()
        )
        if collision:
            continue
        neg_made += 1
        cases.append({
            "case_id": _unique_id(f"NEG-{neg_existing + neg_made:03d}"),
            "type": "negative",
            "query": data["query"].strip(),
            "invented_model": invented,
            "review_status": "pending",
        })
    print(f"   负例新增 {neg_made} 条（尝试 {neg_attempts} 次，碰撞丢弃 {neg_attempts - neg_made}）")

    # ---------- 产物 ----------
    meta = {
        "generated_by": "build_rag_cases.py",
        "model": model,
        "knowledge": sum(1 for c in cases if c["type"] == "knowledge"),
        "product": sum(1 for c in cases if c["type"] == "product"),
        "negative": sum(1 for c in cases if c["type"] == "negative"),
        "failed": skipped,
        "note": "review_status=pending：人工抽检通过后改为 approved；评测脚本只跑 approved",
    }
    out_path.write_text(
        json.dumps({"meta": meta, "cases": cases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    review_lines = ["# RAG 评测集人工抽检清单", "", f"共 {len(cases)} 条（pending 状态）", ""]
    for c in cases:
        expect = c.get("expected_doc", "")
        if c.get("expected_section"):
            expect += f" · {c['expected_section']}"
        if c.get("expected_product_id"):
            expect = c["expected_product_id"]
        if c["type"] == "negative":
            expect = f"应查无结果（编造型号：{c.get('invented_model')}）"
        review_lines.append(f"- [ ] `{c['case_id']}` {c['query']}  →  {expect}")
    (out_path.parent / "rag_cases_review.md").write_text(
        "\n".join(review_lines), encoding="utf-8"
    )

    print(f"\n🎉 评测集完成：{out_path}")
    print(f"   知识 {meta['knowledge']} / 商品 {meta['product']} / 负例 {meta['negative']}，生成失败 {skipped}")
    print(f"   人工抽检清单：{out_path.parent / 'rag_cases_review.md'}")


if __name__ == "__main__":
    main()
