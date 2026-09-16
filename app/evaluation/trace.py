"""运行轨迹：沙箱采集的 Agent 运行过程与结果载体（P6）。

沙箱通过给共享的 OpenAI client、ToolManager、Router 插桩，把一次会话里发生的
所有 LLM 调用（token / 请求的工具 / 延迟）、工具返回、每轮路由与回复都记录到
RunTrace 里。评估器随后只读 RunTrace，不再碰 Agent 内部，实现「采集」与「评分」
解耦。落盘字段与 harness部分.md H11 的事件字段约定对齐（平台无关的事实描述）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMCallRecord:
    """单次 LLM 调用的记录。"""

    purpose: str  # 启发式标注：router / react / answer
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_calls: list[dict] = field(default_factory=list)  # 本次响应请求的工具 [{name, arguments}]
    latency_ms: float = 0.0
    finish_reason: str = ""  # 停止原因：stop=正常；length=输出被截断（回复中途断掉的诊断依据）


@dataclass
class ToolObservation:
    """单次工具调用的输入与返回（数值保真检查的 ground truth 来源）。"""

    name: str
    arguments: dict
    result: str | dict  # execute_tool 的信封（沙箱采集为 dict；判分器兼容两种形态）


@dataclass
class RunTrace:
    """一条用例完整运行的轨迹。"""

    case_id: str
    turns: list[str]
    replies: list[str] = field(default_factory=list)   # 每轮最终回复（与 turns 对齐）
    routes: list[list[str]] = field(default_factory=list)  # 每轮路由结果（与 turns 对齐）
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_observations: list[ToolObservation] = field(default_factory=list)
    # 轮次边界：第 i 项 = 第 i 轮结束时的观测数。len==len(turns) 时可按轮切分
    # 观测序列（第 i 轮调用 = tool_observations[bounds[i-1]:bounds[i]]），
    # 供 process judge 按轮对齐材料（v2 修复"最后一轮问题 vs 全会话调用"错位）
    tool_boundaries: list[int] = field(default_factory=list)
    langfuse_trace_id: str | None = None  # 上报开启时该用例的 Langfuse trace id（判分后挂 score）
    error: str | None = None  # 运行异常信息，None=正常

    # ---------- 便捷聚合属性 ----------
    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.llm_calls)

    @property
    def num_llm_calls(self) -> int:
        return len(self.llm_calls)

    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_observations)

    @property
    def tool_call_names(self) -> list[str]:
        """实际调用过的工具名（按调用顺序）。"""
        return [obs.name for obs in self.tool_observations]

    @property
    def abnormal_llm_calls(self) -> list[LLMCallRecord]:
        """非正常结束的 LLM 调用——输出被截断类问题的诊断入口。

        判据细化（2026-09-16 实测）：`tool_calls` 是 ReAct 请求工具的**正常**结束
        原因（该轮跑批 46/128 次），≠stop 不等于异常；只有 length（预算被吃光/
        输出截断）、content_filter 等才算异常。空字符串 = 未采集到，不判异常。
        """
        return [
            c for c in self.llm_calls
            if c.finish_reason and c.finish_reason not in ("stop", "tool_calls")
        ]

    @property
    def all_replies_text(self) -> str:
        """全部回复拼接（跨轮断言的检查范围）。"""
        return "\n".join(self.replies)

    def to_dict(self) -> dict:
        """完整快照，供 runs/*.json 落盘（字段与 H11 事件约定对齐）。"""
        return {
            "case_id": self.case_id,
            "turns": self.turns,
            "replies": self.replies,
            "routes": self.routes,
            "total_tokens": self.total_tokens,
            "num_llm_calls": self.num_llm_calls,
            "num_tool_calls": self.num_tool_calls,
            "llm_calls": [
                {
                    "purpose": c.purpose,
                    "model": c.model,
                    "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens,
                    "total_tokens": c.total_tokens,
                    "latency_ms": round(c.latency_ms, 1),
                    "finish_reason": c.finish_reason,
                    "requested_tools": c.tool_calls,
                }
                for c in self.llm_calls
            ],
            "tool_calls": [
                {"name": obs.name, "arguments": obs.arguments, "result": obs.result}
                for obs in self.tool_observations
            ],
            "tool_boundaries": self.tool_boundaries,
            "langfuse_trace_id": self.langfuse_trace_id,
            "error": self.error,
        }
