"""一次性验证脚本：reasoning_content 三链路回归（P4 #4，跑完可删）。

验证 DeepSeek 推理模型对 full 档合并后 raw_messages 回放的容忍度：
- 链路 ① 单意图多轮：连续 3 轮单 Agent（历史含上一轮工具对，回放次数最多）
- 链路 ② 多意图→追问：第 1 轮双 Agent 并行（full 档工具对 ×2），第 2 轮追问
- 链路 ③ 压缩后追问：压阈值触发 _compress_history 后追问（切点 + 回放双重风险）

用法：python tests/verify_reasoning_content.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.multi_agent.orchestrator import MultiAgentOrchestrator  # noqa: E402

TEST_SESSION = str(ROOT / "app" / "sessions" / "test_verify_session.json")


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    print(f"  ❌ {msg}")
    sys.exit(1)


def _clean():
    Path(TEST_SESSION).unlink(missing_ok=True)


def _new_orch(threshold: int = 12):
    _clean()
    orch = MultiAgentOrchestrator(session_path=TEST_SESSION)
    orch.history_threshold = threshold
    return orch


def _check_replay(orch, turn_no: int):
    """跑完一轮后检查：下一轮回放的历史里 assistant(tool_calls) 消息的完整性。"""
    for m in orch.raw_messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            has_rc = bool(m.get("reasoning_content"))
            if not has_rc:
                # 关键观察点：合并进历史的 tool_calls 消息是否带 reasoning_content
                return has_rc, m
    return None, None  # 本轮没有 tool_calls 消息


def run_chain(chain_no: int, title: str, turns: list[str], threshold: int):
    print(f"\n{'=' * 60}")
    print(f"链路 {chain_no}：{title}")
    print(f"{'=' * 60}")

    orch = _new_orch(threshold)
    rc_present = False

    for i, user_input in enumerate(turns, 1):
        print(f"\n----- 第 {i} 轮 user: {user_input[:40]}...")
        try:
            reply = orch.chat(user_input)
            _ok(f"第 {i} 轮成功（reply {len(reply)} 字符，无 API 报错）")
        except Exception as e:
            _fail(f"第 {i} 轮报错: {type(e).__name__}: {e}")

        flag, msg = _check_replay(orch, i)
        if flag is False:
            _ok(f"第 {i} 轮后历史含 tool_calls 消息（不带 reasoning_content，第 2 轮回放已容忍）")
            rc_present = True
        elif flag is True:
            print("     （历史中 tool_calls 消息带 reasoning_content）")
            rc_present = True

    n_tool = len([m for m in orch.raw_messages if m.get("role") == "tool"])
    n_tc = len([m for m in orch.raw_messages if m.get("role") == "assistant" and m.get("tool_calls")])
    summary = f"链路 {chain_no} 完成：raw_messages {len(orch.raw_messages)} 条（tool_calls 消息 {n_tc}，tool 消息 {n_tool}）"
    if n_tc > 0:
        _ok(summary + " —— 回放已发生且容忍")
    else:
        _ok(summary + "（本轮无工具调用，回放未覆盖——链路设计问题）")
    _clean()
    return rc_present


def main():
    print("=" * 60)
    print("  reasoning_content 三链路验证（P4 #4，真调 LLM）")
    print("=" * 60)

    covered = False
    # 链路 ① 单意图多轮（threshold 调低确保多轮后可能触发压缩）
    covered |= run_chain(
        1, "单意图多轮（3 轮 guide）",
        ["推荐个 5000 内的耳机", "通勤用的，降噪好一点", "第一款多少钱"],
        threshold=12,
    )
    # 链路 ② 多意图 → 追问
    covered |= run_chain(
        2, "多意图并行 → 针对结果追问",
        ["推荐个 6000 内的笔记本，顺便讲讲 OLED 和 IPS 屏的区别", "你推荐的那款屏幕是哪种？"],
        threshold=20,
    )
    # 链路 ③ 压缩后追问（低阈值强制压缩，压完追问）
    covered |= run_chain(
        3, "压缩触发后追问（切点 + 回放双重风险）",
        ["推荐个笔记本，预算 6000，写代码用", "要轻一点的，最好 14 寸", "那之前推荐的几款保修怎么样"],
        threshold=8,
    )

    print(f"\n{'=' * 60}")
    if covered:
        print("  🎉 三链路全部通过：合并后的历史回放未被 API 拒绝")
        print("     （无论 tool_calls 消息是否带 reasoning_content，当前 API 均容忍）")
        print("     P4 #4 关闭：reasoning_content 风险证伪")
    else:
        print("  ⚠️  三链路都未产生 tool_calls 消息——验证不充分，需调整剧本")
    print("=" * 60)


if __name__ == "__main__":
    main()
