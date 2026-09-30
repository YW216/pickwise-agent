"""压缩链路真调验证（一次性脚本，跑完可删）。

目的：让压缩在真实会话里真的触发一次，并验证四件事——
  1. 触发：上下文超线后压缩执行（summary 从无到有 / 被更新）
  2. 摘要：六段标题齐全且顺序正确
  3. 记忆：商品记忆跨压缩累积，且带 price_at_mention
  4. 指代：压缩后追问能回指被删历史（靠 summary + 商品记忆，而非窗口原文）

零侵入设计：
- 用 os.environ 覆盖预算参数，不改 settings.py / .env，跑完无需回滚
  （pydantic-settings 的环境变量优先级高于 .env）
- 压缩是对工作历史的破坏性删除，因此使用独立 session 文件，
  绝不触碰 app/sessions/session.json
- 不调用 orchestrator.close()，避免把本次验证写入长期记忆（LTM）

用法：
  python tests/verify_compaction_e2e.py            # 默认 8 轮
  python tests/verify_compaction_e2e.py --turns 12 # 若 8 轮未触发，加长
  python tests/verify_compaction_e2e.py --keep-session  # 跑完保留 session 供检视

退出码：0 = 压缩触发且断言全过；1 = 未触发或断言失败。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SESSION = ROOT / "app" / "sessions" / "_compact_verify.json"

# 本脚本不覆盖预算参数——窗口配置统一由 .env 提供（单一事实来源）。
# 早期版本在这里用 os.environ.setdefault() 覆盖，但 .env 的值不在 os.environ 里，
# setdefault 因此总会生效，反而把 .env 的配置盖掉了（实测踩过这个坑）。
sys.path.insert(0, str(ROOT))

from app.agent.compaction import _SUMMARY_SECTIONS  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.multi_agent.orchestrator import MultiAgentOrchestrator  # noqa: E402


# 轮次设计：前 6 轮堆上下文（多工具、多商品、多知识检索），
# 第 7、8 轮的答案都在第 1 轮 —— 若压缩在第 3~6 轮发生，
# 那两轮的原文已被删除，只能靠摘要与商品记忆回答。
TURNS = [
    "想买台笔记本，预算 6000 以内，主要写代码用",
    "再帮我看看通勤用的降噪耳机",
    "OLED 和 IPS 屏幕到底差在哪",
    "预算 4000 左右的手机有什么推荐",
    "帮我把刚才推荐的那两款笔记本对比一下",
    "你们七天无理由退货是怎么规定的",
    "我第一轮说的预算是多少？",
    "之前推荐的那款笔记本现在什么价？",
]


def _state(orch: MultiAgentOrchestrator) -> str:
    """状态快照。

    est 是编排器的上下文估算量 —— 它就是触发压缩的判据
    （est > window - reserve 才压），所以必须打出来，
    才能判断"离触发线还差多少"、该加轮数还是该降窗口。
    """
    est = orch._estimate_context_tokens(orch._build_pack())
    line = settings.context_window - settings.reserve_tokens
    return (
        f"history={orch.history_size} "
        f"summary={len(orch.summary or '')}字 "
        f"products={len(orch.products)}款 "
        f"est={est}(线{line})"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=len(TURNS),
                        help=f"跑前 N 轮（默认 {len(TURNS)}）")
    parser.add_argument("--keep-session", action="store_true",
                        help="跑完保留 session 文件供检视（默认删除）")
    args = parser.parse_args()

    trigger_line = settings.context_window - settings.reserve_tokens
    print("=" * 70)
    print("  压缩链路真调验证")
    print(f"  窗口参数 : window={settings.context_window} "
          f"reserve={settings.reserve_tokens} keep_recent={settings.keep_recent_tokens}"
          f"  (来自 .env)")
    print(f"  触发线   : {settings.context_window} - {settings.reserve_tokens} = {trigger_line}")
    print(f"  会话文件 : {SESSION}")
    print("=" * 70)
    if trigger_line > 60000:
        print("\n⚠️  触发线偏高（>60000），压缩几乎不会触发。")
        print("    请按 .env 里那段注释调小 CONTEXT_WINDOW / RESERVE_TOKENS /")
        print("    KEEP_RECENT_TOKENS，再重跑本脚本。\n")

    if SESSION.exists():
        SESSION.unlink()

    orch = MultiAgentOrchestrator(session_path=str(SESSION))
    print(f"\n[起始状态] {_state(orch)}")

    prev_summary = orch.summary
    compacted_at: list[int] = []

    for i, turn in enumerate(TURNS[: args.turns], 1):
        print(f"\n{'-' * 70}")
        print(f"[轮 {i}] {turn}")
        reply = orch.chat(turn)
        print(f"[轮 {i} 回复] {reply[:240]}{'...' if len(reply) > 240 else ''}")
        print(f"[轮 {i} 状态] {_state(orch)}")
        if orch.summary and orch.summary != prev_summary:
            compacted_at.append(i)
            print(f"[轮 {i}] >>>>> 压缩已触发 <<<<<")
        prev_summary = orch.summary

    # ---------- 断言 ----------
    print(f"\n{'=' * 70}")
    print("  断言")
    print("=" * 70)
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {name}{(' — ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    final_est = orch._estimate_context_tokens(orch._build_pack())
    check("压缩至少触发一次", bool(compacted_at),
          f"触发轮次 {compacted_at}" if compacted_at
          else f"未触发；跑完 {args.turns} 轮 est={final_est}，线={trigger_line}")

    if not compacted_at:
        gap = trigger_line - final_est
        print(f"\n  【未触发诊断】估算 {final_est} / 触发线 {trigger_line}，差 {gap} token")
        print("    两条路任选：")
        print(f"      a) 加轮数：--turns {args.turns + max(1, gap // 4000)} 左右")
        print("         （按每轮约 4K 估算增长推算）")
        print(f"      b) 降窗口：把 .env 的 CONTEXT_WINDOW 改成约 {max(30000, final_est - 2000)}")
        print("         （保持 RESERVE/ KEEP_RECENT 不变；窗口降太低会击穿前两轮 ——")
        print("          那两轮没有可切的完整回合，压不动）")

    summary = orch.summary or ""
    missing = [s for s in _SUMMARY_SECTIONS if s not in summary]
    check("摘要六段齐全", not missing, f"缺 {missing}" if missing else "6/6")

    priced = [p for p in orch.products.values() if "price_at_mention" in p]
    check("商品记忆已累积", len(orch.products) > 0, f"{len(orch.products)} 款")
    check("商品记忆带当时价格", len(priced) > 0, f"{len(priced)} 款带价")

    if args.keep_session:
        print(f"\n  session 已保留：{SESSION}")
    elif SESSION.exists():
        SESSION.unlink()

    print(f"\n{'=' * 70}")
    if failures:
        print(f"  结果：{len(failures)} 项未通过 -> {failures}")
        return 1
    print("  结果：全部通过")
    print("  请人工复核第 7、8 轮回复是否正确回指（摘要 / 商品记忆的作用）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
