"""工具并行收益度量探针（只读，不改业务代码）。

目的：回答"是否值得把 SubAgent 的工具串行循环改成并行执行"。

度量对象（一次 chat 内）：
1. 每个 Agent 每一轮（max_steps 内的一步）发起的 tool_call 数量
2. 每个 tool_call 的执行耗时
3. 该轮 LLM 调用耗时 —— 用来算"工具耗时占该轮总耗时的比例"
4. 整次 chat 的端到端耗时

原理：monkey-patch ToolManager.execute_tool（所有工具的唯一汇聚点）做计时，
     patch SubAgent._complete 记录 LLM 调用耗时，最后聚合打印。

用法：
    # 默认：先清空会话历史，再跑（推荐，保证干净上下文）
    /f/Anaconda/envs/searchagent/python.exe tests/measure_tool_concurrency.py

    # 指定问题（可多个，会依次跑，问题之间自动清历史）
    ... tests/measure_tool_concurrency.py "我要20台笔记本的信息" "推荐续航长的笔记本"

    # 保留会话历史（测"有历史时"的真实表现）
    ... tests/measure_tool_concurrency.py --keep "继续"

    # 连长期记忆一起清（recall_user_memory 会命中旧记忆时用）
    ... tests/measure_tool_concurrency.py --reset-memory

注意：
- 本脚本会真实调用 LLM 与数据库，产生 token 消耗。跑批由用户自行决定。
- 只统计，不改任何业务代码；patch 在进程内生效，脚本退出即恢复。
- 【重要】必须清空会话历史再测，否则模型会复用上一轮的结论、直接跳过工具调用
  （已踩过：历史里有 20 次 get_detail 的记录，模型回复"上一条已经给你了"，
  工具调用数 = 0，度量失效）。
"""

from __future__ import annotations

import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, ".")


# thread-local：子 Agent 跑在独立线程里，用它携带"当前 agent / step"标签，
# 避免多 Agent 并行时记录串号。getattr 兜底：未被 _complete patch 覆盖过的
# 线程（如主线程直接调工具）不应崩。
_TL = threading.local()


# ---------------------------------------------------------------- 采集容器


@dataclass
class ToolCallRecord:
    """单次工具调用记录。"""

    agent: str
    name: str
    duration: float


@dataclass
class LlmCallRecord:
    """单次 LLM 调用记录。"""

    agent: str
    duration: float
    has_tools: bool
    tool_call_count: int


@dataclass
class RoundRecord:
    """一轮工具执行批次（同一轮 LLM 响应内的全部 tool_call）。

    并行的收益只能在轮内兑现，所以这是度量的核心单位：
    串行耗时 = 该轮各次之和；并行耗时 ≈ 该轮各次的最大值。
    """

    agent: str
    step: int
    durations: list[float] = field(default_factory=list)

    @property
    def serial(self) -> float:
        return sum(self.durations)

    @property
    def parallel(self) -> float:
        return max(self.durations) if self.durations else 0.0


@dataclass
class PhaseRecord:
    """端到端的分段耗时。

    串行循环的收益必须放在"Agent 执行"这一段里比较才有意义——
    会话初始化、路由这些开销与工具并行无关，混在一起会稀释结论。
    """

    init: float = 0.0        # Orchestrator 构造（含真值层加载、session 反序列化）
    reset: float = 0.0       # 清空会话历史
    chat: float = 0.0        # 一次 chat 全程
    # chat 内部细分（由 patch 累计）
    router: float = 0.0
    agents: float = 0.0


@dataclass
class Probe:
    """全局采集器。

    注意：Orchestrator 用 ThreadPoolExecutor 并行跑子 Agent，多个 Agent 的
    记录会交错写入。因此所有采集都带 agent 标签，报告按 agent + step 分组，
    跨 Agent 的记录不会被错误合并。
    """

    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    llm_calls: list[LlmCallRecord] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)
    phase: PhaseRecord = field(default_factory=PhaseRecord)

    def reset(self) -> None:
        self.tool_calls.clear()
        self.llm_calls.clear()
        self.rounds.clear()
        self.phase = PhaseRecord()


PROBE = Probe()


# ---------------------------------------------------------------- patch 安装


def install_patches() -> None:
    """安装计时 patch（进程内生效）。

    轮次边界通过包装 SubAgent.handle 内的单次工具执行暴露：handle 每一步会
    连续调用 execute_call_as_message 若干次，我们无法从外部直接看到步边界，
    因此改用 handle 源码里的步进标记——patch `_print_action` 作为"一次工具
    执行即将发生"的信号不可靠，改为在 handle 外层按消息结构切分：
    实际上最稳的做法是直接 patch execute_tool 计时，并由 handle patch 维护
    "当前 step"（通过给 handle 传一个包装过的 max_steps 无法拿到 step 号）。

    最终方案：patch handle 时，把该 agent 的记录收集器替换为局部收集器，
    并利用 `_complete` patch 的调用次数推出 step 号（每个 step 恰好一次
    LLM 调用 + 紧随其后的若干工具调用）。二者通过 (agent, llm_seq) 关联。
    """
    from app.agent.tools.manager import ToolManager
    from app.multi_agent.agents import SubAgent

    # 每个 Agent 的 LLM 调用序号（用于推断 step）
    llm_seq: dict[str, int] = {}
    # 每个 Agent 当前 step 待收集的耗时桶：{(agent, step): RoundRecord}
    pending: dict[tuple[str, int], RoundRecord] = {}

    def _bucket(agent: str, step: int) -> RoundRecord:
        key = (agent, step)
        if key not in pending:
            pending[key] = RoundRecord(agent=agent, step=step)
            PROBE.rounds.append(pending[key])
        return pending[key]

    # ---- patch 1: 工具执行计时（唯一汇聚点）----
    _orig_execute_tool = ToolManager.execute_tool

    def timed_execute_tool(self, name, arguments, seen=None):
        t0 = time.perf_counter()
        try:
            return _orig_execute_tool(self, name, arguments, seen)
        finally:
            dt = time.perf_counter() - t0
            # 归属到"当前 Agent 的当前 step"——由 thread-local 携带。
            # 兜底：若该线程从未经过 _complete（如程序侧直调工具），
            # 以 "?" / 0 记账，不让探针本身抛异常。
            agent = getattr(_TL, "agent", "?")
            step = getattr(_TL, "step", 0)
            rec = _bucket(agent, step)
            rec.durations.append(dt)
            PROBE.tool_calls.append(
                ToolCallRecord(agent=agent, name=name, duration=dt)
            )

    ToolManager.execute_tool = timed_execute_tool

    # ---- patch 2: LLM 调用计时 + 推进 step ----
    _orig_complete = SubAgent._complete

    def timed_complete(self, messages, tools):
        # 进入本次 _complete 即意味着新的一步开始（step 从 1 计）
        n = llm_seq.get(self.name, 0) + 1
        llm_seq[self.name] = n
        _TL.agent = self.name
        _TL.step = n

        t0 = time.perf_counter()
        resp = _orig_complete(self, messages, tools)
        dt = time.perf_counter() - t0
        try:
            n_tools = len(resp.choices[0].message.tool_calls or [])
        except Exception:
            n_tools = 0
        PROBE.llm_calls.append(
            LlmCallRecord(
                agent=self.name, duration=dt,
                has_tools=bool(tools), tool_call_count=n_tools,
            )
        )
        # 该 step 的工具批次在此刻定型（后续 execute_tool 会写入同一桶）
        _bucket(self.name, n)
        return resp

    SubAgent._complete = timed_complete

    # ---- patch 3: 每次 handle 重置该 Agent 的 step 计数 + 累计 Agent 执行耗时 ----
    _orig_handle = SubAgent.handle

    def resetting_handle(self, messages, max_steps=5):
        llm_seq[self.name] = 0
        t0 = time.perf_counter()
        try:
            return _orig_handle(self, messages, max_steps=max_steps)
        finally:
            PROBE.phase.agents += time.perf_counter() - t0

    SubAgent.handle = resetting_handle

    # ---- patch 4: Router 计时 ----
    from app.multi_agent.router import Router

    _orig_route = Router.route

    def timed_route(self, user_input, history, summary=None):
        t0 = time.perf_counter()
        try:
            return _orig_route(self, user_input, history, summary=summary)
        finally:
            PROBE.phase.router += time.perf_counter() - t0

    Router.route = timed_route


# ---------------------------------------------------------------- 报告


def _fmt_ms(sec: float) -> str:
    return f"{sec * 1000:.0f}ms"


def report(e2e_seconds: float) -> None:
    """打印度量报告。"""
    print("\n" + "=" * 64)
    print("  工具并行收益度量报告")
    print("=" * 64)

    tc = PROBE.tool_calls
    lc = PROBE.llm_calls
    rounds_with_tools = [r for r in PROBE.rounds if r.durations]

    print(f"\n本次 chat 端到端耗时：{e2e_seconds:.2f}s")
    print(f"LLM 调用次数：{len(lc)}   工具调用次数：{len(tc)}"
          f"   含工具的轮次：{len(rounds_with_tools)}")

    # ---- 0. 分段耗时（看清时间花在哪）----
    ph = PROBE.phase
    print("\n【0】端到端分段耗时")
    print(f"    路由：        {ph.router:.2f}s")
    print(f"    Agent 执行：  {ph.agents:.2f}s   （工具并行的收益只发生在这一段）")
    other = e2e_seconds - ph.router - ph.agents
    print(f"    其他：        {max(other, 0):.2f}s   "
          f"（压缩检查 / 上下文组装 / 会话保存 等）")
    print(f"    ── 合计      {e2e_seconds:.2f}s")
    if ph.agents > 0:
        tool_share = sum(r.duration for r in tc) / ph.agents * 100
        print(f"    → 工具耗时占 Agent 执行段的 {tool_share:.1f}%"
              f"（这是并行改造能压缩的真实比例）")

    # ---- 1. tool_call 批次大小分布（核心指标）----
    print("\n【1】每轮工具批次的规模分布")
    print("    （并行收益的直接决定因素：批次越大，并行省得越多）")
    if not rounds_with_tools:
        print("    （本次没有工具调用）")
    else:
        from collections import Counter

        sizes = [len(r.durations) for r in rounds_with_tools]
        dist = Counter(sizes)
        for n in sorted(dist):
            print(f"    批次含 {n} 个 tool_call 的轮次：{dist[n]} 次")
        multi = [n for n in sizes if n > 1]
        print(
            f"    → 平均 {statistics.mean(sizes):.2f} 个/批，最大 {max(sizes)} 个"
        )
        print(
            f"    → 可并行收益的批次（>1 个）：{len(multi)}/{len(sizes)}"
            f"  ({len(multi) / len(sizes) * 100:.0f}%)"
        )

    # ---- 2. 单个工具耗时 ----
    print("\n【2】各工具单次执行耗时")
    by_tool: dict[str, list[float]] = {}
    for r in tc:
        by_tool.setdefault(r.name, []).append(r.duration)
    if not by_tool:
        print("    （本次没有工具调用）")
    for name, durs in sorted(by_tool.items(), key=lambda kv: -sum(kv[1])):
        print(
            f"    {name:22s} 调用 {len(durs)} 次  "
            f"合计 {_fmt_ms(sum(durs)):>8s}  "
            f"平均 {_fmt_ms(statistics.mean(durs)):>8s}  "
            f"最大 {_fmt_ms(max(durs)):>8s}"
        )

    # ---- 3. 精确并行收益（按轮分组：串行 sum vs 并行 max）----
    print("\n【3】并行收益（按轮分组，同一轮内的 tool_call 才可并发）")
    if not rounds_with_tools:
        print("    （本次没有工具调用，无并行收益）")
    else:
        serial = sum(r.serial for r in rounds_with_tools)
        parallel = sum(r.parallel for r in rounds_with_tools)
        saved = serial - parallel
        print(f"    当前实现（串行）工具耗时合计：{serial:.2f}s")
        print(f"    改为并行后工具耗时估算：    {parallel:.2f}s")
        print(f"    → 预计节省：{saved:.2f}s"
              f"  （占端到端 {saved / e2e_seconds * 100:.1f}%）")
        print("\n    逐批次明细：")
        for r in sorted(rounds_with_tools, key=lambda x: -x.serial):
            print(
                f"      [{r.agent} 第{r.step}步] "
                f"{len(r.durations)} 次  串行 {_fmt_ms(r.serial)} → "
                f"并行 {_fmt_ms(r.parallel)}  省 {_fmt_ms(r.serial - r.parallel)}"
            )

    # ---- 4. 工具耗时占比 ----
    print("\n【4】工具耗时在端到端中的占比（并行能触及的上界）")
    tool_total = sum(r.duration for r in tc)
    llm_total = sum(r.duration for r in lc)
    print(f"    工具总耗时：{tool_total:.2f}s")
    print(f"    LLM 总耗时：{llm_total:.2f}s")
    if e2e_seconds > 0:
        print(
            f"    → 工具占端到端：{tool_total / e2e_seconds * 100:.1f}%"
        )

    # ---- 5. 结论 ----
    print("\n【5】结论")
    if not rounds_with_tools:
        print("    本次无工具调用，无需并行。")
    else:
        serial = sum(r.serial for r in rounds_with_tools)
        saved = serial - sum(r.parallel for r in rounds_with_tools)
        denom = PROBE.phase.agents or e2e_seconds
        pct = saved / denom * 100 if denom else 0
        print(f"    预计节省 {saved:.2f}s，占 Agent 执行段 {pct:.1f}%")
        if pct < 5:
            print("    判定：<5% —— 不建议做并行改造。")
        elif pct < 15:
            print("    判定：5~15% —— 边际，视实现成本决定。")
        else:
            print("    判定：>15% —— 值得做并行改造。")
        print("    提示：单次 sample 波动大，建议换几个问题各跑一遍再定。")

    print("\n" + "=" * 64)


# ---------------------------------------------------------------- 主流程


def main() -> None:
    args = sys.argv[1:]

    fresh = True
    reset_memory_too = False
    queries: list[str] = []

    for a in args:
        if a in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        elif a == "--keep":
            fresh = False
        elif a == "--fresh":
            fresh = True
        elif a == "--reset-memory":
            # 连长期记忆一起清（否则 recall_user_memory 仍会命中旧记忆，
            # 可能让模型跳过工具调用——与刚踩的 0 调用坑同源）
            reset_memory_too = True
        elif a.startswith("--"):
            print(f"未知参数：{a}（用 --help 查看用法）", file=sys.stderr)
            sys.exit(2)
        else:
            queries.append(a)

    if not queries:
        queries = ["我要20台笔记本的信息"]

    install_patches()
    print("探针已安装（只读，不改业务代码）")

    from app.multi_agent.orchestrator import MultiAgentOrchestrator
    from app.config.settings import settings

    # ---- 三段计时：构造 / 清理 / 执行 ----
    t_init = time.perf_counter()
    agent = MultiAgentOrchestrator()
    PROBE.phase.init = time.perf_counter() - t_init
    print(f"Orchestrator 就绪（耗时 {PROBE.phase.init:.2f}s）")

    if fresh:
        t_rst = time.perf_counter()
        agent.reset()
        if reset_memory_too:
            agent.memory_manager.reset_short_term()
            # 长期记忆文件：直接移走（不删，可人工恢复）
            mem_dir = Path(settings.memory_dir)
            for f in mem_dir.glob("*.json"):
                f.replace(f.with_suffix(".json.disabled"))
        PROBE.phase.reset = time.perf_counter() - t_rst
        print(f"已清空会话历史（耗时 {PROBE.phase.reset:.2f}s）")
    else:
        print(f"保留会话历史（当前 {agent.history_size} 条）")

    print(f"\n待测问题：{queries}\n")

    for q in queries:
        PROBE.reset()
        PROBE.phase.init = 0.0
        PROBE.phase.reset = 0.0
        print("─" * 64)
        print(f"👤 {q}")
        t0 = time.perf_counter()
        try:
            reply = agent.chat(q)
            elapsed = time.perf_counter() - t0
            print(f"🤖 {reply[:200]}{'...' if len(reply) > 200 else ''}")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"⚠️  执行出错：{exc}")
        PROBE.phase.chat = elapsed
        report(elapsed)

        # 每个问题之间隔离历史，否则第二个问题会继承第一个问题的上下文
        if fresh and q is not queries[-1]:
            agent.reset()
            print("（已清空历史，准备下一个问题）")

    agent.close()


if __name__ == "__main__":
    main()
