"""隔离实验统计：对比「投影开 / 关」两组跑批轨迹里的越权工具调用。

要回答的问题只有一个：
    关闭 _project_history 时，模型是否真的会模仿其他 Agent 的工具名去调用，
    从而被白名单拒收（fail「未知工具」）？开启后这个数字是否归零？

口径（对应 AGENT_CONFIGS 的白名单结构）：
- **越权发起**：发起调用的 Agent 的白名单里没有这个工具名。
  判据用 Agent 归属而不是"全局工具表"——get_detail / retrieve_knowledge /
  load_skill 是共享工具，两个 Agent 调都合法，不能算越权。
- **被拒**：结果信封 success=false 且 error 含「未知工具」。这是越权发起的
  下游结果，两者在正常情况下相等；不相等说明执行层行为变了（值得查）。
- **对照组健康度**：leak_control_* 全程同 Agent，两组都必须"有调用、零被拒"。
  若投影组把同 Agent 的调用也折叠掉了，说明投影范围写宽了——那是误伤，
  不是修复成功。

用法（PowerShell）：
    # 两组对比
    python app/scripts/analyze_isolation.py `
        --off app/evaluation/runs_projection_off `
        --on  app/evaluation/runs_projection_on

    # 只看一组
    python app/scripts/analyze_isolation.py --off app/evaluation/runs_projection_off
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.multi_agent.agents import AGENT_CONFIGS  # noqa: E402

REJECT_MARK = "未知工具"


# ---------- 读取与解析 ----------

def load_traces(run_dir: Path) -> list[dict]:
    """读一个跑批目录下的全部轨迹（跳过聚合报告）。"""
    if not run_dir.is_dir():
        raise SystemExit(f"❌ 目录不存在: {run_dir}")
    traces = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name in ("report.json",):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "case_id" not in data:  # 非轨迹文件（如 judge_failures 之类）
            continue
        traces.append(data)
    return traces


def iter_calls(trace: dict):
    """遍历工具调用，兼容两种落盘字段名。

    2026-10-21 之前的轨迹里字段叫 tool_calls；更早的分析脚本假设
    tool_observations（trace.py 的 dataclass 字段名）。两种都认，避免
    因为一次改名让历史数据变成不可读。
    """
    for obs in trace.get("tool_calls") or trace.get("tool_observations") or []:
        yield obs


def parse_result(result) -> dict:
    """结果信封：落盘可能是 dict（沙箱）或 str（旧版/手工构造）。"""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def is_rejected(envelope: dict) -> bool:
    return envelope.get("success") is False and REJECT_MARK in str(envelope.get("error", ""))


def is_unauthorized(agent: str, tool: str) -> bool:
    """该工具是否越了**发起者自己**的白名单。

    agent 为空（旧轨迹无归属 / 单 Agent 路径）时无法判定，返回 False 并在
    汇总里单列"无归属调用"，不让它污染越权结论。
    """
    config = AGENT_CONFIGS.get(agent)
    if config is None:
        return False
    return tool not in config["tools"]


def turn_of(index: int, boundaries: list[int]) -> int:
    """观测下标 → 轮次序号（1 起）。boundaries 缺失时返回 0。"""
    for turn, end in enumerate(boundaries, start=1):
        if index < end:
            return turn
    return 0


# ---------- 汇总 ----------

def summarize(run_dirs) -> dict:
    """汇总一个或多个跑批目录（重复实验时一次传入全部）。

    支持多目录是为了回答「诱发是否稳定」：同一用例跑 5 次会落 5 个目录，
    需要把次数加总再判断，而不是逐目录肉眼比。
    """
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    run_dirs = [Path(d) for d in run_dirs]
    traces: list[dict] = []
    for d in run_dirs:
        traces.extend(load_traces(d))
    per_case_map: dict[str, dict] = {}
    total_calls = 0
    total_tokens = 0
    unauthorized = Counter()   # (agent, tool) -> 次数
    rejected = Counter()       # 被拒工具名 -> 次数
    by_turn = Counter()        # 轮次 -> 越权次数
    unattributed = 0
    rejected_total = 0
    single_turns = 0             # 路由恰为单 consult 的轮次
    consult_calls_single = 0     # 单 consult 轮里的 consult 调用数
    consult_unauth_single = 0    # 其中越权数
    consult_calls_multi = 0      # 多 Agent 轮里的 consult 调用数
    consult_unauth_multi = 0     # 其中越权数

    for trace in traces:
        case_unauthorized = 0
        case_rejected = 0
        case_calls = 0
        boundaries = trace.get("tool_boundaries") or []
        routes = trace.get("routes") or []
        calls = list(iter_calls(trace))
        for index, obs in enumerate(calls):
            name = obs.get("name", "")
            agent = obs.get("agent", "")
            envelope = parse_result(obs.get("result"))
            case_calls += 1
            if not agent:
                unattributed += 1
            elif is_unauthorized(agent, name):
                unauthorized[(agent, name)] += 1
                by_turn[turn_of(index, boundaries)] += 1
                case_unauthorized += 1
            if is_rejected(envelope):
                rejected[name] += 1
                case_rejected += 1
                rejected_total += 1

        # consult 的越权机会分两种轮型，实测结论与直觉相反（2026-10-22 全量）：
        # - 单 consult 轮：用户问纯政策/知识，模型通常直接作答，几乎不检索；
        # - 多 Agent 轮：用户明确要「再搜/列一下」，consult 看到 presale 的历史
        #   工具名，反而更可能自己上手搜——它与 presale 并行执行，并不知道
        #   presale 已经把检索做掉了。
        # 两种必须分开报：只拿「单 consult 轮」当分母会把真正的高发窗口
        # 从统计里剔掉（3 次越权全部落在多 Agent 轮）。
        case_single_turns = 0
        case_calls_single = 0
        case_unauth_single = 0
        case_calls_multi = 0
        case_unauth_multi = 0
        for turn_index, route in enumerate(routes):
            start = boundaries[turn_index - 1] if turn_index > 0 else 0
            end = (boundaries[turn_index]
                   if turn_index < len(boundaries) else len(calls))
            single = list(route) == ["consult"]
            if single:
                case_single_turns += 1
            for obs in calls[start:end]:
                if obs.get("agent") != "consult":
                    continue
                bad = is_unauthorized("consult", obs.get("name", ""))
                if single:
                    case_calls_single += 1
                    case_unauth_single += 1 if bad else 0
                else:
                    case_calls_multi += 1
                    case_unauth_multi += 1 if bad else 0

        single_turns += case_single_turns
        consult_calls_single += case_calls_single
        consult_unauth_single += case_unauth_single
        consult_calls_multi += case_calls_multi
        consult_unauth_multi += case_unauth_multi

        total_calls += case_calls
        total_tokens += trace.get("total_tokens", 0) or 0
        # 同一 case_id 可能出现在多个重复目录里：按 id 累加，等价于"跑 N 次的总和"
        row = per_case_map.setdefault(trace.get("case_id", "?"), {
            "case_id": trace.get("case_id", "?"),
            "runs": 0,
            "calls": 0,
            "unauthorized": 0,
            "rejected": 0,
            "effective_turns": 0,
            "consult_calls_single": 0,
            "consult_unauth_single": 0,
            "consult_calls_multi": 0,
            "consult_unauth_multi": 0,
            "tokens": 0,
            "errors": [],
        })
        row["runs"] += 1
        row["calls"] += case_calls
        row["unauthorized"] += case_unauthorized
        row["rejected"] += case_rejected
        row["effective_turns"] += case_single_turns
        row["consult_calls_single"] += case_calls_single
        row["consult_unauth_single"] += case_unauth_single
        row["consult_calls_multi"] += case_calls_multi
        row["consult_unauth_multi"] += case_unauth_multi
        row["tokens"] += trace.get("total_tokens", 0) or 0
        if trace.get("error"):
            row["errors"].append(trace["error"])

    per_case = sorted(per_case_map.values(), key=lambda r: r["case_id"])

    return {
        "dir": " + ".join(str(d) for d in run_dirs),
        "cases": len(traces),
        "total_calls": total_calls,
        "total_tokens": total_tokens,
        "unauthorized_total": sum(unauthorized.values()),
        "unauthorized": unauthorized,
        "rejected_total": rejected_total,
        "rejected": rejected,
        "unattributed": unattributed,
        "by_turn": by_turn,
        "effective_turns": single_turns,
        "consult_calls_single": consult_calls_single,
        "consult_unauth_single": consult_unauth_single,
        "consult_calls_multi": consult_calls_multi,
        "consult_unauth_multi": consult_unauth_multi,
        "per_case": per_case,
    }


def control_health(summary: dict) -> tuple[int, int]:
    """对照组（leak_control_*）：有调用且零被拒的用例数 / 对照组总数。"""
    controls = [c for c in summary["per_case"] if c["case_id"].startswith("leak_control")]
    healthy = sum(1 for c in controls if c["calls"] > 0 and c["rejected"] == 0)
    return healthy, len(controls)


# ---------- 输出 ----------

def print_single(s: dict, label: str) -> None:
    print("=" * 78)
    print(f"  {label}")
    print(f"  目录: {s['dir']}")
    print("=" * 78)
    print(f"  会话数            : {s['cases']}")
    print(f"  工具调用总数      : {s['total_calls']}")
    print(f"  越权发起          : {s['unauthorized_total']}")
    print(f"  被拒（未知工具）  : {s['rejected_total']}")
    print(f"  consult 调用分布  : 单 consult 轮 {s['consult_calls_single']} 次"
          f"（越权 {s['consult_unauth_single']}） / "
          f"多 Agent 轮 {s['consult_calls_multi']} 次"
          f"（越权 {s['consult_unauth_multi']}）")
    print(f"  单 consult 轮次   : {s['effective_turns']} 轮")
    print(f"  无 Agent 归属调用 : {s['unattributed']}"
          f"{'   ← 旧轨迹，无法判定越权' if s['unattributed'] else ''}")
    print(f"  总 token          : {s['total_tokens']:,}")

    if s["unauthorized"]:
        print("\n  越权明细（Agent → 工具）:")
        for (agent, tool), count in s["unauthorized"].most_common():
            print(f"    {agent:<10} {tool:<22} {count} 次")
    if s["by_turn"]:
        print("\n  按轮次分布（0 = 边界缺失）:")
        for turn, count in sorted(s["by_turn"].items()):
            print(f"    第 {turn} 轮: {count} 次")
    if s["rejected"]:
        print("\n  被拒工具:")
        for tool, count in s["rejected"].most_common():
            print(f"    {tool:<22} {count} 次")

    healthy, controls = control_health(s)
    print(f"\n  对照组健康度      : {healthy}/{controls}"
          f"{' ✅' if healthy == controls else ' ⚠️ 投影可能误伤同 Agent 轨迹'}")


def print_per_case(off: dict, on: dict) -> None:
    """逐用例对照：越权次数 off → on。只看有越权或轨迹异常的用例。"""
    off_map = {c["case_id"]: c for c in off["per_case"]}
    on_map = {c["case_id"]: c for c in on["per_case"]}
    rows = []
    for case_id in sorted(set(off_map) | set(on_map)):
        a = off_map.get(case_id, {})
        b = on_map.get(case_id, {})
        ua, ub = a.get("unauthorized", 0), b.get("unauthorized", 0)
        ra, rb = a.get("rejected", 0), b.get("rejected", 0)
        ea = "；".join(a.get("errors") or [])
        eb = "；".join(b.get("errors") or [])
        if ua or ub or ra or rb or ea or eb:
            rows.append((case_id, a.get("runs", 0), ua, ub, ra, rb, ea, eb))

    if not rows:
        print("\n  没有任何越权/被拒记录——两组都干净（或都没诱发出来）。")
        return
    print("\n" + "=" * 78)
    print("  逐用例对照（越权次数 / 被拒次数）")
    print("=" * 78)
    print(f"  {'用例':<30}{'跑次':<6}{'越权 off→on':<16}{'被拒 off→on':<16}异常")
    for case_id, runs, ua, ub, ra, rb, ea, eb in rows:
        anomaly = "；".join(x for x in (ea, eb) if x)[:26]
        print(f"  {case_id:<30}{runs:<6}{f'{ua} → {ub}':<16}{f'{ra} → {rb}':<16}{anomaly}")


def print_compare(off: dict, on: dict) -> None:
    print("\n" + "=" * 78)
    print("  对照结论")
    print("=" * 78)
    print(f"  {'指标':<22}{'关投影(off)':>14}{'开投影(on)':>14}")
    print(f"  {'会话数':<22}{off['cases']:>14}{on['cases']:>14}")
    print(f"  {'工具调用总数':<22}{off['total_calls']:>14}{on['total_calls']:>14}")
    print(f"  {'越权发起':<22}{off['unauthorized_total']:>14}{on['unauthorized_total']:>14}")
    print(f"  {'被拒(未知工具)':<22}{off['rejected_total']:>14}{on['rejected_total']:>14}")
    print(f"  {'单 consult 轮次':<21}{off['effective_turns']:>14}{on['effective_turns']:>14}")
    print(f"  {'consult 越权(单轮)':<20}{off['consult_unauth_single']:>14}"
          f"{on['consult_unauth_single']:>14}")
    print(f"  {'consult 越权(多Agent轮)':<18}{off['consult_unauth_multi']:>14}"
          f"{on['consult_unauth_multi']:>14}")
    token_line = (
        f"  {'总 token':<22}{off['total_tokens']:>14,}{on['total_tokens']:>14,}"
    )
    if off["total_tokens"]:
        delta = (on["total_tokens"] / off["total_tokens"] - 1) * 100
        token_line += f"   ({delta:+.1f}%)"
    print(token_line)

    healthy_off, controls = control_health(off)
    healthy_on, _ = control_health(on)
    print(f"  {'对照组健康度':<20}{f'{healthy_off}/{controls}':>14}{f'{healthy_on}/{controls}':>14}")

    print()
    if off["unauthorized_total"] > 0 and on["unauthorized_total"] == 0:
        print("  ✅ 机制成立：关投影出现越权，开投影归零。")
    elif off["unauthorized_total"] == 0 and on["unauthorized_total"] == 0:
        print("  ⚠️ 两组都是 0：本批没有诱发泄漏（看路由是否真切到 consult、"
              "或样本量/温度是否需要加大），不能据此判定修复有效。")
    elif on["unauthorized_total"] > 0:
        print("  ❌ 开投影仍有越权：投影覆盖不全（检查混调块是否整块保留）。")
    else:
        print("  ❌ 开投影比关投影更差：轨迹异常，先查 error 字段。")

    if off["unattributed"] or on["unattributed"]:
        print("  ⚠️ 存在无 agent 归属的旧轨迹，越权判定不完整——重跑一次再统计。")


def main() -> None:
    parser = argparse.ArgumentParser(description="隔离实验：投影开/关越权对比")
    parser.add_argument(
        "--off", required=True, nargs="+",
        help="关投影组的 runs 目录（可传多个：重复实验的各次目录，统计时加总）",
    )
    parser.add_argument(
        "--on", default=None, nargs="+",
        help="开投影组的 runs 目录（省略则只分析 --off）",
    )
    args = parser.parse_args()

    off = summarize(args.off)
    print_single(off, "关投影（对照组：历史原样暴露）")
    if not args.on:
        print_per_case(off, {"per_case": []})
        return

    on = summarize(args.on)
    print()
    print_single(on, "开投影（生产行为：越权块已折叠）")
    print_per_case(off, on)
    print_compare(off, on)


if __name__ == "__main__":
    main()
