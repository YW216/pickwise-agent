"""评测上报器：把沙箱运行与判分结果上报 Langfuse（P6 平台化，批次 2）。

职责边界（分层原则，见 11-评估.md）：
- 只做「上报」，不参与判分——判分仍完全由 metrics/judges 负责（Langfuse 是仓库不是裁判）；
- Langfuse 未安装 / 未配 KEY / 上报异常 → enabled=False，全链路 no-op，本地评测零影响；
- 概念映射：trace = 一条用例的运行，session = 一次跑批（run_id），
  observation = LLM 调用（drop-in 自动）/ 工具执行（本模块显式上报），
  score = 判分结果（e2e_pass / judge_* / tools_hit_rate）。

langfuse 4.x API 要点（已核 4.15.3 源码，勿按旧版文档写）：
- drop-in 只认 langfuse_prompt / langfuse_public_key 两个前缀参数；trace 归组与
  session 走 OTel 上下文传播，不通过 create() 的 kwarg（避免未知参数泄漏给真 API）；
- propagate_attributes(session_id/tags/trace_name) 设置 trace 级属性；
- start_as_current_observation(as_type="span") 建用例 span；drop-in 的 generation
  与工具 span 会自动挂到"当前 span"下（实测同 trace，无需注入 trace_id）；
- create_score(trace_id=...) 把判分结果挂回用例 trace；
- flush() 保证进程退出前异步上报落地。
"""

from __future__ import annotations

import contextlib
import logging
import os

logger = logging.getLogger(__name__)


class LangfuseReporter:
    """Langfuse 上报器；不可用时自动降级为 no-op（调用方无需判空）。"""

    def __init__(self, run_id: str, enabled: bool = True):
        self.run_id = run_id
        self._client = self._build_client() if enabled else None
        self.enabled = self._client is not None

    @classmethod
    def disabled(cls) -> "LangfuseReporter":
        """no-op 实例：离线自检/未配置上报时使用。"""
        return cls(run_id="", enabled=False)

    @staticmethod
    def _build_client():
        """构造 Langfuse 客户端；缺依赖/缺 KEY 返回 None（静默降级 + 一行警告）。"""
        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            logger.info("未配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY，评测上报关闭")
            return None
        try:
            from langfuse import Langfuse

            return Langfuse()
        except Exception as e:  # noqa: BLE001 —— 上报是旁路，任何失败都不该影响评测
            logger.warning("Langfuse 客户端构造失败，上报关闭: %s", e)
            return None

    # ---------- 上报接口 ----------

    @contextlib.contextmanager
    def case_scope(self, case_id: str, category: str):
        """用例作用域：进入后沙箱内所有 LLM/工具调用自动归入同一 trace。

        yield 该用例的 trace_id（上报关闭时为 None），供后续挂 score。
        """
        if not self.enabled:
            yield None
            return
        from langfuse import propagate_attributes

        with propagate_attributes(
            session_id=self.run_id, tags=[case_id, category], trace_name=case_id,
        ):
            with self._client.start_as_current_observation(
                name=f"eval:{case_id}",
                as_type="span",
                input={"case_id": case_id, "category": category},
            ) as span:
                yield span.trace_id

    @contextlib.contextmanager
    def tool_span(self, name: str, arguments: dict):
        """单次工具调用 span：input=调用参数；调用方拿到 span 后 update(output=结果)。"""
        if not self.enabled:
            yield None
            return
        with self._client.start_as_current_observation(
            name=f"tool.{name}", as_type="span", input=arguments,
        ) as span:
            yield span

    @contextlib.contextmanager
    def judge_scope(self, trace_id: str | None, aspect: str):
        """judge 调用作用域：挂回**用例 trace**（remote parent），标签 judge:<aspect>。

        为什么要显式挂：langfuse 的 OpenAI 补丁是 wrapt 类级补丁、全局生效——
        judge 的 client 同样被 instrument，而 judge 在 case_scope 之外调用；
        不处理会在看板上产生大量独立根 trace（20 条用例 × 2 aspect 的噪音）。
        挂回用例 trace 后，判分过程与 Agent 轨迹同 trace：可审计、无污染。
        """
        if not (self.enabled and trace_id):
            yield None
            return
        with self._client.start_as_current_observation(
            name=f"judge:{aspect}", as_type="span",
            trace_context={"trace_id": trace_id},
        ) as span:
            yield span

    @staticmethod
    def is_instrumented(client) -> bool:
        """该 client 是否已被 langfuse OpenAI 补丁包装（wrapt BoundFunctionWrapper 标记）。

        用于判断能否安全注入 name= 等 Langfuse 专用 kwarg——未包装时注入会漏给真 API。
        """
        try:
            return type(client.chat.completions.create).__name__ == "BoundFunctionWrapper"
        except Exception:  # noqa: BLE001
            return False

    def report_case(self, trace_id, *, passed: bool, checks, judge_scores,
                    tool_hits, error: str | None = None) -> None:
        """把一条用例的判分结果作为 score 挂回其 trace（失败静默，绝不影响评测）。"""
        if not (self.enabled and trace_id):
            return
        try:
            failures = [c.name for c in checks if c.applicable and not c.passed]
            if error:
                comment = f"运行异常: {error}"
            elif failures:
                comment = f"断言失败: {'、'.join(failures)}"
            else:
                comment = "全部断言通过"
            self._client.create_score(
                trace_id=trace_id, name="e2e_pass", value=passed,
                data_type="BOOLEAN", comment=comment,
            )
            for aspect, judged in (judge_scores or {}).items():
                if judged and isinstance(judged.get("score"), int):
                    self._client.create_score(
                        trace_id=trace_id,
                        name=f"judge_{aspect}",
                        value=judged["score"],
                        data_type="NUMERIC",
                        comment="；".join(judged.get("reasons") or [])[:500],
                    )
            if tool_hits and tool_hits[1]:
                self._client.create_score(
                    trace_id=trace_id, name="tools_hit_rate",
                    value=tool_hits[0] / tool_hits[1], data_type="NUMERIC",
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Langfuse score 上报失败（忽略）: %s", e)

    def trace_url(self, trace_id: str | None) -> str | None:
        """用例 trace 的网页地址（给终端报告提供直达链接）。"""
        if not (self.enabled and trace_id):
            return None
        try:
            return self._client.get_trace_url(trace_id=trace_id)
        except Exception:  # noqa: BLE001
            return None

    def flush(self) -> None:
        """进程退出前落地上报（异步批量）。"""
        if not self.enabled:
            return
        try:
            self._client.flush()
        except Exception as e:  # noqa: BLE001
            logger.warning("Langfuse flush 失败（忽略）: %s", e)
