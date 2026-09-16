"""子 Agent 定义：每个子 Agent 有专属的 system prompt 和工具子集。

SubAgent 封装了一个轻量级 ReAct 循环，由 Orchestrator 调度执行。
"""

import json

from openai import OpenAI

from app.agent.context_budget import (
    ContextOverflowError,
    estimate_messages_tokens,
    estimate_tool_definitions_tokens,
    is_context_overflow,
    prepare_messages,
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
        system_prompt: str,
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
        self.system_prompt = system_prompt
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
        working = prepare_messages(messages, self.tool_result_max_chars)

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
                content = assistant_msg.content or ""
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
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)

                self._print_action(func_name, func_args)
                # bug #4 修复：execute_tool 返回 dict，直接塞进 tool 消息 content
                # 违反 OpenAI 协议（要求 str）；统一走信封序列化
                result_str = self.tool_manager.execute_tool_as_message(
                    func_name, func_args,
                )
                self._print_observation(result_str)

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                }
                new_messages.append(tool_msg)
                working.append({
                    **tool_msg,
                    "content": tool_result_view(
                        result_str, self.tool_result_max_chars,
                    ),
                })

        # 兜底策略：超过最大步数时不给工具，让模型基于已有观察给出最终回答
        try:
            response = self._complete(working, [])
        except Exception as exc:
            if not isinstance(exc, ContextOverflowError) and not is_context_overflow(exc):
                raise
            # 溢出同样上抛：编排器压缩后整体重试一次（策略与循环内一致，见 handle 注释）
            raise ContextOverflowError(str(exc)) from exc
        content = response.choices[0].message.content or ""
        new_messages.append({"role": "assistant", "content": content})
        return content, new_messages

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

    def _print_action(self, func_name: str, func_args: dict) -> None:
        args_str = ", ".join(f"{k}={v!r}" for k, v in func_args.items())
        print(f"  🔧 [{self.name}·调用工具] {func_name}({args_str})")

    def _print_observation(self, result: str) -> None:
        display = result if len(result) <= 300 else result[:300] + "..."
        print(f"  📋 [{self.name}·工具结果] {display}")
