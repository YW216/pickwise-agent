"""LLM 事实提取：从对话中抽取短期/长期记忆事实。

模式与 app/agent/summarizer.py 一致：格式化对话 → 调用 LLM → 解析结果。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from openai import OpenAI

from app.prompts.memory import LTM_EXTRACTION_PROMPT, STM_EXTRACTION_PROMPT

if TYPE_CHECKING:
    from app.agent.memory.long_term import MemoryFact

# 一次抽取最多沉淀几条画像：长对话容易一次吐出十几条，直接把上限刷满。
MAX_NEW_FACTS_PER_RUN = 3


def _build_transcript(messages: list[dict]) -> str:
    """将消息列表格式化为文本摘要（复用 summarizer 的逻辑）。"""
    lines = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""

        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    lines.append(f"客服：[调用工具 {name}]")
            if content:
                lines.append(f"客服：{content}")
        elif role == "tool":
            display = content if len(content) <= 200 else content[:200] + "..."
            lines.append(f"[工具结果] {display}")

    return "\n".join(lines)


async def extract_short_term_facts(
    client: OpenAI,
    model: str,
    recent_messages: list[dict],
    existing_facts: list[str],
) -> list[str]:
    """从最近对话中提取/更新短期记忆事实。"""
    transcript = _build_transcript(recent_messages)
    if not transcript.strip():
        return existing_facts

    existing_text = "\n".join(f"- {f}" for f in existing_facts) if existing_facts else "（暂无）"
    prompt = STM_EXTRACTION_PROMPT.format(existing_facts=existing_text)

    response = await client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
    )
    raw = response.choices[0].message.content.strip()

    if "无新信息" in raw:
        return existing_facts

    new_facts = [line.strip().lstrip("- ") for line in raw.splitlines() if line.strip()]
    if not new_facts:
        return existing_facts

    merged = list(existing_facts)
    existing_lower = {f.lower() for f in merged}
    for fact in new_facts:
        if fact.lower() not in existing_lower:
            merged.append(fact)
            existing_lower.add(fact.lower())
    return merged


async def extract_long_term_facts(
    client: OpenAI,
    model: str,
    messages: list[dict],
    summary: str | None,
    existing_facts: list,
) -> tuple[list[MemoryFact], str]:
    """从完整会话中提取用户画像 + 交互摘要（一次最多 MAX_NEW_FACTS_PER_RUN 条）。"""
    from app.agent.memory.long_term import MemoryFact, PROFILE_SLOTS

    parts = []
    if summary:
        parts.append(f"【对话摘要】\n{summary}")

    transcript = _build_transcript(messages)
    if transcript:
        parts.append(f"【对话内容】\n{transcript}")

    if not parts:
        return [], ""

    existing_text = (
        "\n".join(
            f"- [{f.slot}] {f.key} = {_value_text(f.value)}" for f in existing_facts
        )
        if existing_facts
        else "（暂无）"
    )
    prompt = LTM_EXTRACTION_PROMPT.format(existing_ltm=existing_text)

    response = await client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
    )
    raw = response.choices[0].message.content.strip()

    data = _parse_json(raw)
    if data is None:
        return [], "（提取失败）"

    now = datetime.now().date().isoformat()
    new_facts = []
    for item in data.get("profile") or []:
        if not isinstance(item, dict):
            continue
        slot = str(item.get("slot", "")).strip()
        key = str(item.get("key", "")).strip()
        value = item.get("value", "")
        if isinstance(value, str):
            value = value.strip()
        if isinstance(value, list):
            value = [str(part).strip() for part in value if str(part).strip()]
        # 只认固定四类，且 key/value 必须非空——结构不全宁可丢。
        if slot not in PROFILE_SLOTS or not key or not value:
            continue
        new_facts.append(MemoryFact(
            slot=slot,
            key=key,
            value=value,
            quote=str(item.get("quote", "")).strip(),
            updated_at=now,
        ))
        if len(new_facts) >= MAX_NEW_FACTS_PER_RUN:
            break  # 一次会话最多沉淀 3 条，防止长对话把画像刷满

    interaction_summary = str(data.get("interaction_summary", "") or "").strip()
    return new_facts, interaction_summary


def _value_text(value) -> str:
    return " > ".join(map(str, value)) if isinstance(value, list) else str(value)


def _parse_json(raw: str):
    """容错解析：模型偶尔会包 ```json 代码块或前后带说明。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
