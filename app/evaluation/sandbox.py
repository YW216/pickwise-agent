"""评估沙箱：隔离、可复现地重跑测试集，并采集运行全过程（第9期）。

沙箱做三件事：
1. 隔离：每条用例独立的临时 session 文件；关闭记忆读写（否则 default.json 会注入
   prompt 污染评分）；关闭 MCP 只用本地 mock 工具（保证可复现，且让幻觉检测有确定
   的 ground truth）。
2. 插桩：单/多 Agent 都只共享一个 OpenAI client 实例，给它的
   chat.completions.create / beta.chat.completions.parse 打补丁，即可捕获整个会话
   所有 LLM 调用的 token、被请求的工具、延迟——无需改动 chat.py / orchestrator.py。
   工具返回值则通过包裹 ToolManager.execute_tool 采集。
3. 执行：顺序跑完用例的多轮输入，把过程与结果填进 RunTrace 返回。
4. 上报（批次 2，可选）：reporter 开启时整条用例包在 Langfuse 用例 span 里——
   drop-in 采集的 LLM 调用与这里显式上报的工具 span 自动归入同一 trace；
   上报关闭（未配 KEY/离线）时全链路 no-op。

关键：绝不调用 agent.close()（会触发长期记忆巩固的 LLM 写入，污染且烧钱）；
所有补丁在 finally 中还原。
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from app.config.settings import settings
from app.evaluation.dataset import EvalCase
from app.evaluation.reporter import LangfuseReporter
from app.evaluation.trace import LLMCallRecord, RunTrace, ToolObservation


class Sandbox:
    """Agent 评估沙箱：构建隔离环境、插桩采集、跑用例产出 RunTrace。"""

    def __init__(self, mode: str = "multi", tmp_root: str | None = None,
                 reporter: LangfuseReporter | None = None):
        # mode 保留参数以兼容旧调用方；single（旧客服 EcomAgent）已随 chat.py 删除
        self.mode = mode
        self.tmp_root = Path(tmp_root) if tmp_root else Path(tempfile.mkdtemp(prefix="eval_sandbox_"))
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        # 上报器：默认 no-op（离线自检/未配置时零影响）
        self.reporter = reporter or LangfuseReporter.disabled()
        # drop-in 是否已给 client 打补丁（决定能否注入 name= 等 Langfuse 专用参数）
        self._lf_naming = False
        # 是否在 Router 作用域内（供 purpose 标注；由 _wrap_route 置位）
        self._in_router = False

    def session_path_for(self, case_id: str) -> str:
        return str(self.tmp_root / f"{case_id}.json")

    def _build_agent(self, session_path: str):
        """在隔离配置下构建被测 Agent。"""
        # 记忆：启用（对齐生产——recall_user_memory 需真实可调，否则"记忆驱动"
        # 类用例的工具断言永远失败），但 memory_dir 指向每条用例独立的沙箱目录：
        # 既不读真实用户记忆（可复现），也隔离用例间状态。MCP 仍关闭（幻觉
        # 检测需要确定的 ground truth）。
        settings.memory_enabled = True
        settings.memory_dir = str(Path(session_path).with_suffix("")) + "_memory"
        settings.mcp_enabled = False

        from app.multi_agent.orchestrator import MultiAgentOrchestrator
        return MultiAgentOrchestrator(session_path=session_path)

    def run(self, case: EvalCase) -> RunTrace:
        """跑一条用例，返回采集到的运行轨迹。"""
        trace = RunTrace(case_id=case.id, turns=list(case.turns))
        trace._pending_route: list[str] = []  # 当前轮路由暂存（run 循环收割后清空）
        session_path = self.session_path_for(case.id)

        agent = None
        patches: list[tuple] = []  # (obj, attr, original) 供还原
        with self.reporter.case_scope(case.id, case.category) as trace_id:
            trace.langfuse_trace_id = trace_id
            try:
                agent = self._build_agent(session_path)
                self._instrument(agent, trace, patches)

                for turn in case.turns:
                    reply = agent.chat(turn)
                    trace.replies.append(reply)
                    trace.routes.append(list(trace._pending_route or []))
                    trace._pending_route = []
                    # 轮次边界：本轮结束时的观测数，供 process judge 按轮切分调用序列
                    trace.tool_boundaries.append(len(trace.tool_observations))

            except Exception as e:  # noqa: BLE001 —— 单条用例异常不应中断整轮评估
                trace.error = f"{type(e).__name__}: {e}"
            finally:
                for obj, attr, original in patches:
                    setattr(obj, attr, original)
                if agent is not None:
                    self._close_tool_managers(agent)
                # 注意：刻意不调用 agent.close()，避免长期记忆巩固写入

        return trace

    # ---------- 插桩 ----------
    def _instrument(self, agent, trace: RunTrace, patches: list[tuple]) -> None:
        """给共享 client、各 ToolManager、（多 Agent）Router 打补丁。"""
        # drop-in 是否已给该 client 打补丁（wrapt 类级补丁，全局生效）——决定能否
        # 安全注入 name= 等 Langfuse 专用参数（未打补丁时注入会漏给真 API 报错）
        self._lf_naming = bool(self.reporter.enabled) and self.reporter.is_instrumented(
            agent.client
        )

        # 1) LLM client：create + beta.parse
        completions = agent.client.chat.completions
        patches.append((completions, "create", completions.create))
        completions.create = self._wrap_create(completions.create, trace)

        beta_completions = agent.client.beta.chat.completions
        patches.append((beta_completions, "parse", beta_completions.parse))
        beta_completions.parse = self._wrap_parse(beta_completions.parse, trace)

        # 2) 工具执行
        for tm in self._tool_managers(agent):
            patches.append((tm, "execute_tool", tm.execute_tool))
            tm.execute_tool = self._wrap_execute_tool(tm.execute_tool, trace)

        # 3) 多 Agent 路由
        if self.mode == "multi" and hasattr(agent, "router"):
            patches.append((agent.router, "route", agent.router.route))
            agent.router.route = self._wrap_route(agent.router.route, trace)

    def _wrap_create(self, original, trace: RunTrace):
        def wrapper(*args, **kwargs):
            purpose = self._guess_purpose(kwargs, in_router=self._in_router)
            # 给 Langfuse drop-in 传 name（看板里显示 agent:router/react/answer，
            # 否则清一色 "OpenAI-generation"）；仅在补丁确实生效时才传，避免漏给真 API
            if self._lf_naming:
                kwargs.setdefault("name", f"agent:{purpose}")
            start = time.time()
            response = original(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000
            self._record_llm_call(trace, response, latency_ms, purpose=purpose)
            return response
        return wrapper

    def _wrap_parse(self, original, trace: RunTrace):
        def wrapper(*args, **kwargs):
            if self._lf_naming:
                kwargs.setdefault("name", "agent:extract")
            start = time.time()
            response = original(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000
            self._record_llm_call(trace, response, latency_ms, purpose="extract")
            return response
        return wrapper

    def _wrap_execute_tool(self, original, trace: RunTrace):
        def wrapper(name: str, arguments: dict) -> str:
            # 工具执行上报为独立 span（input=参数，output=结果信封）——当前 trace
            # 由 run() 的 case_scope 决定；上报关闭时 span 为 None，纯旁路
            with self.reporter.tool_span(name, dict(arguments)) as span:
                result_str = original(name, arguments)
                if span is not None:
                    span.update(output=result_str)
            trace.tool_observations.append(
                ToolObservation(name=name, arguments=dict(arguments), result=result_str)
            )
            return result_str
        return wrapper

    def _wrap_route(self, original, trace: RunTrace):
        def wrapper(*args, **kwargs):
            # 进入 Router 作用域：其内部的 LLM 调用据此标为 router（见 _guess_purpose）
            self._in_router = True
            try:
                scenarios = original(*args, **kwargs)
            finally:
                self._in_router = False
            trace._pending_route = list(scenarios)
            return scenarios
        return wrapper

    # ---------- 辅助 ----------
    @staticmethod
    def _guess_purpose(kwargs: dict, in_router: bool = False) -> str:
        """标注 LLM 调用用途（报告/看板可读，不作硬断言）。

        判据改为**调用位置驱动**（2026-09-16 二次修正）：不再依赖 max_tokens 数值——
        该启发式已被预算变更打破两次（max_tokens==10 过时 → 改用 ≤1024 又因 router
        预算 512→2048 再次把 router 误标为 answer）。现在：
        - router：在 Router.route() 作用域内发生的调用（由 _wrap_route 置标志）
        - react：带 tools 的子 Agent ReAct 循环
        - answer：其余（最终回复 / Result 整合 / 摘要等其他生成调用）
        """
        if in_router:
            return "router"
        if kwargs.get("tools"):
            return "react"
        return "answer"

    @staticmethod
    def _record_llm_call(trace: RunTrace, response, latency_ms: float, purpose: str) -> None:
        usage = getattr(response, "usage", None) #获取token使用情况
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        tool_calls: list[dict] = []
        finish_reason = ""
        try:
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", "") or ""
            message = choice.message
            for tc in (getattr(message, "tool_calls", None) or []):
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })
        except (AttributeError, IndexError):
            pass

        model = getattr(response, "model", "") or ""
        trace.llm_calls.append(LLMCallRecord(
            purpose=purpose,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        ))

    def _tool_managers(self, agent) -> list:
        if self.mode == "multi" and hasattr(agent, "agents"):
            return [a.tool_manager for a in agent.agents.values()]
        if hasattr(agent, "tool_manager"):
            return [agent.tool_manager]
        return []

    def _close_tool_managers(self, agent) -> None:
        for tm in self._tool_managers(agent):
            try:
                tm.close()
            except Exception:  # noqa: BLE001
                pass
