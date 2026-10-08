"""子 Agent 定义：每个子 Agent 有专属的 system prompt 和工具子集。

SubAgent 封装了一个轻量级 ReAct 循环，由 Orchestrator 调度执行。
"""

from openai import OpenAI

from app.agent.context_budget import (
    ContextOverflowError,
    estimate_messages_tokens,
    estimate_tool_definitions_tokens,
    is_context_overflow,
    tool_result_view,
)
from app.agent.tools.manager import ToolManager
from app.prompts.agents import (
    CONSULT_MULTI_PROMPT,
    CONSULT_PROMPT,
    PRESALE_MULTI_PROMPT,
    PRESALE_PROMPT,
)


AGENT_CONFIGS = {
    "presale": {
        "name": "PickWise-售前",
        "prompts": {"single": PRESALE_PROMPT, "multi": PRESALE_MULTI_PROMPT},
        "tools": {  # 8 个（6 业务 + load_skill + recall_user_memory）
            # 2026-09-11：search_by_reviews 暂缓（search_products 接位）
            # 2026-09-15：retrieve_reviews 摘除（评价入库暂缓、本地评价数据下线）
            "search_catalog", "get_user_favorites", "search_products",
            "get_detail", "compare_products",
            "retrieve_knowledge",
            "load_skill", "recall_user_memory",
        },
    },
    "consult": {
        "name": "PickWise-咨询",
        "prompts": {"single": CONSULT_PROMPT, "multi": CONSULT_MULTI_PROMPT},
        # 3 个（政策规则问答由 retrieve_knowledge 的政策文档承担；2026-09-15
        # retrieve_warranty 摘除——保修条款数据下线）
        "tools": {"retrieve_knowledge", "get_detail", "load_skill"},
    },
}


class SubAgent:
    """专业子 Agent：拥有独立的 prompt 和工具子集，执行 ReAct 循环。"""

    def __init__(
        self,
        name: str,
        tool_manager: ToolManager,
        client: OpenAI,
        model: str,
        temperature: float,
        tool_result_max_chars: int = 12000,
        context_window: int = 200000,
        reserve_tokens: int = 16384,
        reasoning_effort: str = "",
    ):
        self.name = name
        self.tool_manager = tool_manager
        self.client = client
        self.model = model
        self.temperature = temperature
        self.tool_result_max_chars = tool_result_max_chars
        self.context_window = context_window
        self.reserve_tokens = reserve_tokens
        self.reasoning_effort = reasoning_effort  # 空 = 不传，用服务端默认
        self.last_prompt_tokens: int | None = None
        self.last_estimated_tokens = 0

    def handle(
        self, messages: list[dict], max_steps: int = 5,
    ) -> tuple[str, list[dict]]:
        """执行 ReAct 循环，返回 (最终文本, 新增消息列表)。

        并行安全约定（设计文档 6.4）：内部只操作 working 私有副本（Copy-on-Write），
        绝不触碰 Orchestrator 的 raw_messages；new_messages 是记账式收集的增量通道，
        最后一条恒为最终答复的 assistant 消息（设计文档 10.5 位置约定）。
        """
        new_messages: list[dict] = []
        # 私有副本（Copy-on-Write，绝不触碰 Orchestrator 的 raw_messages）：
        # ① 新建 list——handle 内部会 append，不能改到调用方传来的列表；
        # ② 每条再浅拷贝一层——防止将来有人就地改写元素时污染上游。
        # 注：tool 消息在入库时已是视图，这里不做任何截断。
        working = [dict(message) for message in messages]
        # 本轮各工具签名的调用次数（熔断用）。生命周期 = 一次 handle，所以跨轮
        # 不会误伤——用户下一轮问同样的问题，仍然照常执行。
        seen: dict[str, int] = {}

        for i in range(max_steps):
            print(f"第 {i+1} 步·[{self.name}]")
            try:
                response = self._complete(working, self.tool_manager.tool_definitions)
            except Exception as exc:
                if not isinstance(exc, ContextOverflowError) and not is_context_overflow(exc):
                    raise
                # 溢出（预检或 API 溢出）在任意步统一上抛：编排器压缩后整体重试一次。
                # 不就地压缩——溢出的是私有 working，压共享 raw_messages 救不了本次调用；
                # 不带轨迹重入——当前工具全为只读查询，整体重放无正确性风险
                # （写工具接入时重启策略随 H8 重新评估）。
                raise ContextOverflowError(str(exc)) from exc
            assistant_msg = response.choices[0].message

            # 思考内容无论是否有 tool 调用都打印（ReAct 过程可视化）
            if assistant_msg.content:
                self._print_thought(assistant_msg.content)

            if not assistant_msg.tool_calls:
                content = self._final_text(response)
                msg = {"role": "assistant", "content": content}
                new_messages.append(msg)
                return content, new_messages

            msg_dict = {
                "role": "assistant",
                "content": assistant_msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ],
            }

            new_messages.append(msg_dict)
            working.append(msg_dict)

            for tc in assistant_msg.tool_calls:
                # 解析 → 校验 → 熔断 → 执行 → 序列化，全部收口在 ToolManager
                # （三道闸门见其方法说明）。参数用模型给的原始文本打印：
                # 连"参数是坏 JSON"这种情形也能如实显示。
                result_str = self.tool_manager.execute_call_as_message(
                    tc.function.name, tc.function.arguments, seen,
                )
                
                self._print_action(tc.function.name, tc.function.arguments)
                self._print_observation(result_str)

                # 入库即截断：存下来的就是模型看到的（单一真相）。
                # 不再"全量入账 + 发送前另做一份视图"，两者也就不会再漂移。
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result_view(
                        result_str, self.tool_result_max_chars,
                    ),
                }
                new_messages.append(tool_msg)
                working.append(tool_msg)

        # 兜底策略：超过最大步数时不给工具，让模型基于已有观察给出最终回答
        try:
            response = self._complete(working, [])
        except Exception as exc:
            if not isinstance(exc, ContextOverflowError) and not is_context_overflow(exc):
                raise
            # 溢出同样上抛：编排器压缩后整体重试一次（策略与循环内一致，见 handle 注释）
            raise ContextOverflowError(str(exc)) from exc
        content = self._final_text(response)
        new_messages.append({"role": "assistant", "content": content})
        return content, new_messages

    @staticmethod
    def _final_text(response) -> str:
        """HTTP 成功不等于任务成功：空正文/截断/未完成调用不能当最终答复。"""
        choice = response.choices[0]
        content = choice.message.content or ""
        if (
            not isinstance(content, str) or not content.strip()
            or getattr(choice, "finish_reason", None) == "length"
            or getattr(choice.message, "tool_calls", None)
        ):
            raise ValueError("模型未返回完整有效的最终答复")
        return content

    def _complete(self, messages: list[dict], tools: list[dict]):
        """每次 LLM 请求前检查完整输入，并记录真实 usage 供估算校准。"""
        self.last_estimated_tokens = (
            estimate_messages_tokens(messages) + estimate_tool_definitions_tokens(tools)
        )
        if self.last_estimated_tokens > self.context_window - self.reserve_tokens:
            raise ContextOverflowError("请求上下文超过预算")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.reserve_tokens,
            **({"tools": tools} if tools else {}),
            **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}),
        )
        self.last_prompt_tokens = getattr(getattr(response, "usage", None), "prompt_tokens", None)
        return response

    def _print_thought(self, text: str) -> None:
        print(f"\n  💭 [{self.name}·思考] {text}")

    def _print_action(self, func_name: str, raw_arguments: str) -> None:
        """打印工具调用：参数直接用模型给的原始文本（坏 JSON 时也能如实显示）。"""
        print(f"  🔧 [{self.name}·调用工具] {func_name}({raw_arguments})")

    def _print_observation(self, result: str) -> None:
        display = result if len(result) <= 300 else result[:300] + "..."
        print(f"  📋 [{self.name}·工具结果] {display}")
