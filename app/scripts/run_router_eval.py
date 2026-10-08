"""Router 专业评测：逐条评测 + 汇总报告。

评测流程（参考第 9 期 evaluation 模块风格）：
1. 加载用例集（app/evaluation/router_cases.json）
2. 逐条真调 Router.route(input, history)，与期望场景列表完全匹配判定
3. 输出：逐条明细 → 总准确率 → 场景级精确率/召回率/F1（多标签）→
   分维度（group）→ 分难度（difficulty）→ 失败分析
4. 可选 --report 导出 Markdown 报告

用法：python app/scripts/run_router_eval.py [--cases-file ...] [--report out.md]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.evaluation.router_dataset import load_router_cases  # noqa: E402
from app.multi_agent.router import Router, SCENARIO_ORDER  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Router 路由准确率专业评测")
    parser.add_argument("--cases-file", default=str(ROOT / "app" / "evaluation" / "router_cases.json"))
    parser.add_argument("--report", default=None, help="Markdown 报告输出路径（可选）")
    parser.add_argument(
        "--min-accuracy", type=float, default=None,
        help="准确率门槛（如 0.98）；低于门槛退出码 1，供 CI 门禁使用",
    )
    args = parser.parse_args()

    cases = load_router_cases(args.cases_file)
    client = OpenAI(
        api_key=settings.openai_api_key, base_url=settings.openai_base_url,
        timeout=settings.openai_timeout, max_retries=settings.openai_max_retries,
    )
    router = Router(client, settings.model_name)

    # ---------- 逐条评测 ----------
    results = []
    for c in cases:
        try:
            actual = router.route(c.input, c.history or None, summary=c.summary)
        except Exception as e:  # noqa: BLE001 评测兜底：单条失败不中断整体
            actual = [f"ERROR:{e}"]
        results.append({"case": c, "actual": actual, "ok": actual == c.expected})

    lines: list[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit(f"Router 专业评测报告")
    emit(f"用例集: {Path(args.cases_file).name}（{len(cases)} 条）")
    emit(f"模型: {settings.model_name} | 时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    emit("=" * 72)

    # ---------- 逐条明细 ----------
    emit("\n## 一、逐条明细")
    for r in results:
        c = r["case"]
        mark = "PASS" if r["ok"] else "FAIL"
        history_note = f"（history {len(c.history)} 条）" if c.history else ""
        emit(f"[{mark}] {c.id} [{c.group}/{c.difficulty}] {c.description} {history_note}")
        emit(f"      输入: {c.input!r} → 期望 {c.expected} 实际 {r['actual']}")

    # ---------- 汇总统计 ----------
    n, n_ok = len(results), sum(r["ok"] for r in results)
    emit("\n## 二、汇总")
    emit(f"完全匹配准确率: {n_ok}/{n} = {n_ok / n:.1%}")

    # ---------- 场景级多标签指标 ----------
    emit("\n## 三、场景级指标（多标签：精确率 / 召回率 / F1）")
    stats = {s: {"exp": 0, "act": 0, "hit": 0} for s in SCENARIO_ORDER}
    for r in results:
        exp_set, act_set = set(r["case"].expected), set(r["actual"])
        for s in SCENARIO_ORDER:
            if s in exp_set:
                stats[s]["exp"] += 1
            if s in act_set:
                stats[s]["act"] += 1
            if s in exp_set and s in act_set:
                stats[s]["hit"] += 1
    emit(f"  {'场景':<10} {'召回(exp→act)':<16} {'精确(act→exp)':<16} {'F1':<6}")
    for s in SCENARIO_ORDER:
        st = stats[s]
        recall = st["hit"] / st["exp"] if st["exp"] else None
        prec = st["hit"] / st["act"] if st["act"] else None
        f1 = 2 * prec * recall / (prec + recall) if (prec and recall) else None
        emit(f"  {s:<10} {recall if recall is None else f'{recall:.0%}':<16} "
             f"{prec if prec is None else f'{prec:.0%}':<16} "
             f"{f1 if f1 is None else f'{f1:.2f}':<6}  ({st['exp']} 例)")

    # ---------- 分维度 / 分难度 ----------
    def group_stats(key: str) -> dict[str, list[bool]]:
        out: dict[str, list[bool]] = {}
        for r in results:
            out.setdefault(getattr(r["case"], key), []).append(r["ok"])
        return out

    emit("\n## 四、分维度（group）")
    for g, ok_list in sorted(group_stats("group").items()):
        emit(f"  {g:<16} {sum(ok_list)}/{len(ok_list)} = {sum(ok_list) / len(ok_list):.0%}")

    emit("\n## 五、分难度（difficulty）")
    for d, ok_list in sorted(group_stats("difficulty").items()):
        emit(f"  {d:<10} {sum(ok_list)}/{len(ok_list)} = {sum(ok_list) / len(ok_list):.0%}")

    # ---------- 失败分析 ----------
    failures = [r for r in results if not r["ok"]]
    emit(f"\n## 六、失败分析（{len(failures)} 条）")
    for r in failures:
        c = r["case"]
        emit(f"  ✗ {c.id} [{c.group}/{c.difficulty}] {c.description}")
        emit(f"      期望 {c.expected} → 实际 {r['actual']}")

    # ---------- 报告导出 ----------
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text("\n".join(lines), encoding="utf-8")
        print(f"\n报告已导出: {args.report}")

    # ---------- 退出码（批次 3：CI 门禁前置）----------
    # 0 = 全部通过；1 = 有失败用例。当前基线是 100%，任何回归都会立刻让 CI 变红。
    accuracy = n_ok / n if n else 0.0
    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        print(f"\n❌ 准确率 {accuracy:.1%} 低于门槛 {args.min_accuracy:.0%}")
        sys.exit(1)
    if n_ok != n:
        print(f"\n❌ 存在失败用例：{n - n_ok} 条（见上失败分析）")
        sys.exit(1)
    print(f"\n✅ 全部通过（准确率 {accuracy:.1%}）")
    sys.exit(0)


if __name__ == "__main__":
    main()
