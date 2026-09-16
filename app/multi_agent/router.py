"""意图路由器：分析用户消息，输出需要处理的场景列表。

输出契约（设计文档 5.1）：返回场景列表 list[str]，
长度 1 = 单 Agent 路径（N=1 特例），>1 = 多 Agent 并行（P3 起生效）。
保证：非空、元素合法、固定顺序、去重、最多 3 个。
"""

from typing import Optional

from openai import OpenAI

from app.prompts.agents import ROUTER_PROMPT

# 固定优先级顺序，替代旧版无序 set 遍历（修复多值输出命中顺序随机、不可复现的问题）
SCENARIO_ORDER = ["presale", "consult"]
DEFAULT_SCENARIO = "consult"


class Router:
    """使用 LLM 对用户意图分类，输出 1-3 个场景词的有序列表。"""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def route(
        self, user_input: str,
        history: Optional[list[dict]] = None,
        summary: Optional[str] = None,
    ) -> list[str]:
        """返回场景列表（有序、去重、合法、非空）。

        空 / 全非法的 LLM 输出兜底为 [DEFAULT_SCENARIO]。

        Args:
            user_input：当前用户消息
            history：完整对话历史（内部只取最近 4 条 user/assistant 拼上下文）
            summary：历史压缩摘要。压缩后老历史从 raw_messages 消失，
                跨轮指代（"我上次问的那个"）靠摘要兜底，可选
        """
        context_parts = []
        recent_context = self._build_recent_context(history)
        if recent_context:
            context_parts.append(recent_context)
        if summary:
            context_parts.append(f"【此前对话摘要】\n{summary}")
        context = "\n".join(context_parts)

        prompt = ROUTER_PROMPT.format(user_input=user_input)
        if context:
            prompt = context + "\n" + prompt

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            # 推理模型（deepseek-v4-flash）的 reasoning_content 与 content 共享
            # max_tokens 预算：预算太小会被思考吃光导致正文为空（实测 max_tokens=20
            # 时 content 恒为 ''，路由 100% 走兜底）。512 足够容纳思考 + 场景词输出
            max_tokens=512,
        )

        raw = (response.choices[0].message.content or "").strip()
        return self._parse(raw)

    def _parse(self, raw: str) -> list[str]:
        """解析 LLM 原始输出 → 合法场景列表。

        规则（设计文档 5.2）：小写化、兼容中文逗号、去重、
        按 SCENARIO_ORDER 固定顺序排列、空/全非法兜底 DEFAULT_SCENARIO。
        """
        tokens = [t.strip().lower() for t in raw.replace("，", ",").split(",")]
        valid = {t for t in tokens if t in SCENARIO_ORDER}
        if not valid:
            return [DEFAULT_SCENARIO]
        return [s for s in SCENARIO_ORDER if s in valid]

    def _build_recent_context(self, history: Optional[list[dict]]) -> str:
        """取最近对话拼成轻量上下文（沿用旧版行为，帮助多轮指代消解）。"""
        if not history:
            return ""

        recent = [
            m for m in history[-4:]
            if m.get("role") in ("user", "assistant")
        ]
        lines = []
        for m in recent:
            role = "用户" if m["role"] == "user" else "助手"
            content = m.get("content", "")
            if content and len(content) < 200:
                lines.append(f"{role}: {content}")
        if not lines:
            return ""
        return "\n最近对话：\n" + "\n".join(lines) + "\n"
