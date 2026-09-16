"""离线运行端到端评估（P6 第一步：规则判分跑批）。

用法：
  # 全量跑批（真调 LLM，本地 mock 工具；约 13 条 × 数次调用，成本约 1 毛）
  python app/scripts/run_eval.py

  # 单条调试
  python app/scripts/run_eval.py --case-id detail_price

  # 判分器自检：用编造的"回复含假 ID / 假价格"轨迹验证判分器能查出来
  python app/scripts/run_eval.py --self-test

产出：
  - 终端报告：概览矩阵（用例 × 路由/工具/结果断言/judge/token）+ 分类通过率
    + 汇总（通过率百分比 / token 分布 / judge 均分 / 断言全貌）+ 逐条断言明细
  - app/evaluation/runs/<case_id>.json ：每条用例的 RunTrace 落盘
    （字段与 harness部分.md H11 事件约定对齐——将来接平台的数据资产）
  - app/evaluation/runs/report.json   ：聚合报告（含 tool_hits 命中摘要）
  - Langfuse（配置了 LANGFUSE_* 时自动开启）：一次跑批 = 一个 session，
    每条用例一个 trace（含 LLM/工具 span 与判分 score）；--no-report 可关闭
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# 把 .env 写进 os.environ——pydantic 的 env_file 只读进 settings 对象，
# 而 Langfuse SDK 只认 os.environ（LANGFUSE_HOST / PUBLIC_KEY / SECRET_KEY）
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from app.config.settings import settings  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.evaluation.evaluator import Evaluator  # noqa: E402
from app.evaluation.reporter import LangfuseReporter  # noqa: E402
from app.evaluation.sandbox import Sandbox  # noqa: E402

RUNS_DIR = ROOT / "app" / "evaluation" / "runs"


def _self_test() -> bool:
    """判分器自检：构造编造数据的轨迹，断言必须能查出来（判分器也要被测）。"""
    from app.evaluation.dataset import EvalCase
    from app.evaluation.metrics import run_checks
    from app.evaluation.trace import RunTrace, ToolObservation

    case = EvalCase(
        id="self_test", category="自检", description="判分器自检",
        turns=["推荐个 1500 内的降噪耳机"],
        expected_route=["presale"], expected_tools=["search_catalog"],
        reply_prices_within={"category": "耳机", "max_price": 1500},
        reply_prices_in_catalog=True,
        valid_ids_only=True,
    )
    bad = RunTrace(
        case_id="self_test", turns=case.turns,
        replies=["推荐 幻影 X9（LP-99），售价 ¥9999，性价比超高"],
        routes=[["presale"]],
        tool_observations=[ToolObservation(
            name="search_catalog", arguments={}, result="{}",
        )],
    )
    checks = run_checks(case, bad)
    failed = {c.name for c in checks if c.applicable and not c.passed}
    expected_fail = {"valid_ids", "prices_in_catalog"}
    ok = expected_fail <= failed
    print(f"  编造 ID(LP-99) 与价格(9999) → 判定失败的断言: {sorted(failed)}")
    print(f"  自检{'通过 ✅' if ok else '失败 ❌（判分器漏检！）'}")
    return ok


def _fmt_judge(score: dict | None, declared: bool) -> str:
    """judge 单元格：未声明 → "—"；声明但解析失败 → "未判"；正常 → "n/5"。"""
    if not declared:
        return "—"
    if score is None:
        return "未判"
    return f"{score['score']}/5"


def _print_overview(report) -> None:
    """概览矩阵：用例 × 分层（结果 / 路由 / 工具 / 结果断言 / judge / token）。

    列与三层断言同构——哪列先出 ✗，问题就在哪层；逐条明细负责解释为什么。
    """
    print("\n" + "=" * 78)
    print("  概览（用例 × 分层）")
    print("=" * 78)
    print(
        f"  {'用例':<28}{'结果':<5}{'路由':<5}{'工具':<9}"
        f"{'结果断言':<11}{'质量':<7}{'过程':<7}{'token':>10}"
    )
    for r in report.results:
        if r.error:
            print(f"  {r.case_id:<28} ❌ 运行异常: {r.error[:40]}")
            continue
        checks = {c.name: c for c in r.checks if c.applicable}
        route = checks.get("route")
        tools = checks.get("tools")
        result_checks = [c for n, c in checks.items() if n not in ("route", "tools")]
        failed = [c for c in result_checks if not c.passed]

        mark = "✅" if r.passed else "❌"
        route_cell = "—" if route is None else ("✓" if route.passed else "✗")
        if tools is None:
            tools_cell = "—"
        elif r.tool_hits:
            tools_cell = f"{'✓' if tools.passed else '✗'} {r.tool_hits[0]}/{r.tool_hits[1]}"
        else:  # expected_tools=[]（要求零调用）等场景：只看通过与否
            tools_cell = "✓" if tools.passed else "✗"
        if not result_checks:
            result_cell = "—"
        elif failed:
            result_cell = f"✗ {len(failed)}项挂"
        else:
            result_cell = f"✓ {len(result_checks)}项"
        quality = _fmt_judge(r.judge_scores.get("answer_quality"), "answer_quality" in r.judge_scores)
        process = _fmt_judge(r.judge_scores.get("process"), "process" in r.judge_scores)
        tokens = f"{r.trace.total_tokens:,}" if r.trace else "—"
        print(
            f"  {r.case_id:<28}{mark:<5}{route_cell:<5}{tools_cell:<9}"
            f"{result_cell:<11}{quality:<7}{process:<7}{tokens:>10}"
        )


def _print_report(report) -> None:
    _print_overview(report)

    print("\n" + "=" * 78)
    print("  分类通过率")
    print("=" * 78)
    for cat, (passed, total) in report.category_summary().items():
        print(f"  {cat:<8} {passed}/{total}")

    # 汇总：通过率百分比 / token 分布 / judge 均分 / 断言全貌
    total = len(report.results)
    passed = report.passed_count
    traces = [r.trace for r in report.results if r.trace]
    total_tokens = sum(t.total_tokens for t in traces)
    all_checks = [c for r in report.results for c in r.checks if c.applicable]
    failed_checks = [c for c in all_checks if not c.passed]
    print("\n" + "=" * 78)
    print("  汇总")
    print("=" * 78)
    if total:
        print(f"  通过率        : {passed}/{total}  ({passed / total * 100:.0f}%)")
    if traces:
        print(
            f"  总 token 消耗 : {total_tokens:,}"
            f"（平均每用例 {total_tokens // len(traces):,}，"
            f"最大 {max(t.total_tokens for t in traces):,}）"
        )
    for aspect, label in (("answer_quality", "质量"), ("process", "过程")):
        scores = [
            r.judge_scores[aspect]["score"]
            for r in report.results
            if r.judge_scores.get(aspect) is not None
        ]
        if scores:
            print(f"  judge {label}均分 : {sum(scores) / len(scores):.1f}/5（覆盖 {len(scores)} 条）")
    if all_checks:
        print(f"  断言全貌      : {len(all_checks) - len(failed_checks)}/{len(all_checks)} 项检查通过")

    # 截断诊断：finish_reason 非正常结束的调用（tool_calls 是 ReAct 正常原因，不算异常）
    abnormal = [
        (r.case_id, r.trace.abnormal_llm_calls)
        for r in report.results if r.trace and r.trace.abnormal_llm_calls
    ]
    if abnormal:
        total = sum(len(calls) for _, calls in abnormal)
        print(f"  ⚠️  非正常结束的 LLM 调用: {total} 次")
        for cid, calls in abnormal:
            for c in calls:
                print(f"       - {cid} · {c.purpose} · completion={c.completion_tokens} "
                      f"· finish_reason={c.finish_reason}")
        print("     length=输出被截断（回复中途断掉 / 决策预算被思考吃光）")

    print("\n" + "=" * 78)
    print("  逐条明细")
    print("=" * 78)
    for r in report.results:
        mark = "✅" if r.passed else "❌"
        print(f"\n  {mark} {r.case_id}（{r.category}）{r.description}")
        if r.error:
            print(f"     运行异常: {r.error}")
            continue
        for c in r.checks:
            if not c.applicable:
                continue
            mark = "✓" if c.passed else "✗"
            print(f"     [{mark}] {c.name}: {c.detail}")
        for aspect, score in r.judge_scores.items():  # judge 分 advisory，独立展示
            if score is None:
                print(f"     [◇] judge/{aspect}: 未判（解析失败）")
            else:
                print(f"     [◇] judge/{aspect}: {score['score']}/5 —— {'；'.join(score['reasons'][:2])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端评估跑批（规则判分）")
    parser.add_argument(
        "--dataset", default=str(ROOT / "app" / "evaluation" / "cases.json"),
        help="用例集 JSON 路径",
    )
    parser.add_argument("--case-id", default=None, help="只跑指定用例（调试用）")
    parser.add_argument(
        "--self-test", action="store_true",
        help="判分器自检：验证编造数据会被查出来，不跑真实用例",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="不落盘 runs/ 与 report.json",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="关闭 Langfuse 上报（离线/不想产生云端数据时用）",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("  PickWise 端到端评估（规则判分，LLM 仅被测对象）")
    print(f"  被测模型: {settings.model_name} | 时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    if args.self_test:
        sys.exit(0 if _self_test() else 1)

    cases = load_dataset(args.dataset)
    if args.case_id:
        cases = [c for c in cases if c.id == args.case_id]
        if not cases:
            print(f"❌ 找不到用例: {args.case_id}")
            sys.exit(1)
    print(f"\n[1/3] 用例集: {len(cases)} 条")

    # 上报器：一次跑批 = 一个 session（run_id），每条用例一个 trace
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}"
    reporter = LangfuseReporter(run_id=run_id, enabled=not args.no_report)
    print(
        f"       Langfuse 上报: {'开启（session=' + run_id + '）' if reporter.enabled else '关闭（未配置 KEY 或 --no-report）'}"
    )

    print("\n[2/3] 沙箱逐条运行（真调 LLM，mock 工具，独立会话）...")
    from openai import OpenAI
    # judge 的 client 也会被 langfuse 的类级补丁接到（wrapt 全局补丁）：评测侧
    # 不主动为其建 trace，而是由 evaluator 把 judge 调用挂回对应用例 trace 下
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    evaluator = Evaluator(
        sandbox=Sandbox(reporter=reporter), client=client,
        model=settings.model_name, reporter=reporter,
    )
    report = evaluator.run_all(cases)

    print("\n[3/3] 生成报告...")
    _print_report(report)

    if not args.no_save:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        # 清理已下线用例的陈旧轨迹：全量跑批时删除不属于本轮用例集的旧文件，
        # 否则后续分析会把历史文件当成本轮结果（2026-09-16 误判过一次）
        if not args.case_id:
            keep = {f"{r.case_id}.json" for r in report.results if r.trace}
            stale = [
                p for p in RUNS_DIR.glob("*.json")
                if p.name not in keep and p.name != "report.json"
            ]
            for p in stale:
                p.unlink()
            if stale:
                print(f"   清理陈旧轨迹 {len(stale)} 个: {', '.join(p.stem for p in stale)}")
        for r in report.results:
            if r.trace is None:
                continue
            path = RUNS_DIR / f"{r.case_id}.json"
            path.write_text(
                json.dumps(r.trace.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        report_path = RUNS_DIR / "report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n   轨迹与报告已落盘: {RUNS_DIR}")

    if reporter.enabled:
        reporter.flush()  # 异步批量上报，退出前落地
        first = next((r.trace.langfuse_trace_id for r in report.results if r.trace), None)
        url = reporter.trace_url(first)
        print(f"   Langfuse session: {run_id}（Sessions 页可看本次跑批全部用例）")
        if url:
            print(f"   示例 trace: {url}")

    print("\n🎉 评估完成。")
    sys.exit(0 if report.passed_count == len(report.results) else 1)


if __name__ == "__main__":
    main()
