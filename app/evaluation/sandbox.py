"""评估沙箱：隔离、可复现地重跑测试集，并采集运行全过程（第9期）。

沙箱做三件事：
1. 隔离：每条用例独立的临时 session 文件；关闭记忆读写（否则 default.json 会注入
   prompt 污染评分）；关闭 MCP 只用本地 mock 工具（保证可复现，且让幻觉检测有确定
   的 ground truth）。
2. 插桩：单/多 Agent 都只共享一个 OpenAI client 实例，给它的
   chat.completions.create / beta.chat.completions.parse 打补丁，即可捕获整个会话
   所有 LLM 调用的 token、被请求的工具、延迟——无需改动 chat.py / orchestrator.py。
   工具返回值通过包裹 ToolManager.execute_tool 采集；Router.route 采集每轮路由；
   SubAgent.handle 用于把用例的 OTel 上下文重挂进工作线程（见 _wrap_handle——
   线程池会切断 contextvars，不重挂则工具 span 与子 Agent 的 generation 会被
   langfuse 直接跳过）。
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
        # 用例 span 的 OTel 上下文（供工作线程重挂；由 _capture_otel_context 置位）
        self._otel_ctx = None

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

                for turn_index, turn in enumerate(case.turns):
                    reply = agent.chat(turn)
                    trace.replies.append(reply)
                    trace.runtime_failures.extend(
                        {**failure, "turn_index": turn_index}
                        for failure in getattr(agent, "last_failures", [])
                    )
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
                    # 只关闭本用例拥有的 HTTP client，不触发 agent.close() 的记忆写入。
                    close_client = getattr(getattr(agent, "client", None), "close", None)
                    if close_client is not None:
                        try:
                            close_client()
                        except Exception as exc:
                            trace.runtime_failures.append({
                                "stage": "client_close",
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                                "affects_answer": False,
                            })
                # 注意：刻意不调用 agent.close()，避免长期记忆巩固写入

        return trace

    # ---------- 插桩 ----------
    def _instrument(self, agent, trace: RunTrace, patches: list[tuple]) -> None:
        """给共享 client、各 ToolManager、Router、子 Agent 执行入口打补丁。"""
        # drop-in 是否已给该 client 打补丁（wrapt 类级补丁，全局生效）——决定能否
        # 安全注入 name= 等 Langfuse 专用参数（未打补丁时注入会漏给真 API 报错）
        self._lf_naming = bool(self.reporter.enabled) and self.reporter.is_instrumented(
            agent.client
        )

        # 0) 线程上下文：记下用例 span 的 OTel 上下文，供工作线程重挂（见 _wrap_handle）
        self._otel_ctx = self._capture_otel_context()

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

        # 4) 子 Agent 执行入口：工作线程内重挂 OTel 上下文。
        #    编排器无条件用 ThreadPoolExecutor（N=1 也走线程池），而 OTel 的"当前
        #    span"由 contextvars 承载——跨线程不传递（实测工作线程内
        #    get_current_span() 返回 INVALID_SPAN）。langfuse 拿不到活动 span 时会
        #    **直接跳过**该 observation，导致工具 span 与子 Agent 的 generation 全丢。
        for sub in getattr(agent, "agents", {}).values():
            patches.append((sub, "handle", sub.handle))
            sub.handle = self._wrap_handle(sub.handle)

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
        # 签名必须透传 *args/**kwargs，不能写死位置参数：execute_tool 在
        # 2026-09-21「统一信封 + 同签名熔断」后多了第三个参数 seen，
        # 而 Agent 侧一律走 execute_tool_as_message → 按位置传 3 个。
        # 插桩是隐式耦合点（pytest 收不到），写死签名会让整层评测静默失效：
        # 每次工具调用抛 TypeError → 全部用例进 trace.error → 判分全空。
        def wrapper(name: str, arguments: dict, *args, **kwargs):
            # 工具执行上报为独立 span（input=参数，output=结果信封）——当前 trace
            # 由 run() 的 case_scope 决定；上报关闭时 span 为 None，纯旁路
            with self.reporter.tool_span(name, dict(arguments)) as span:
                result_str = original(name, arguments, *args, **kwargs)
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

    # ---------- 线程上下文（评测侧补齐，不改产品代码） ----------

    @staticmethod
    def _capture_otel_context():
        """捕获当前（用例）span 的 OTel 上下文，供工作线程重挂。

        纯旁观能力：未装 langfuse / opentelemetry、或当前没有有效 span 时返回 None，
        调用方退化为直通，评测结果不受任何影响。
        """
        try:
            from opentelemetry import trace as otel_trace_api

            span = otel_trace_api.get_current_span()
            if not span.get_span_context().is_valid:
                return None
            return otel_trace_api.set_span_in_context(span)
        except Exception:  # noqa: BLE001 —— 上报是旁路，任何失败都不该影响评测
            return None

    def _wrap_handle(self, original):
        """把子 Agent 的 handle() 包在用例的 OTel 上下文里执行。

        目的：让工作线程内产生的 LLM 调用（drop-in 自动捕获）与工具 span 都能找到
        "当前 span"，从而挂到用例 trace 上。`attach` 的效果限于当前线程（底层是
        contextvars），因此多场景并行时每个工作线程各自 attach，互不干扰。
        """
        def wrapper(*args, **kwargs):
            if self._otel_ctx is None:
                return original(*args, **kwargs)
            from opentelemetry import context as otel_context_api

            token = otel_context_api.attach(self._otel_ctx)
            try:
                return original(*args, **kwargs)
            finally:
                otel_context_api.detach(token)
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
