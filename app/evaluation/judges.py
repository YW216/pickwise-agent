"""LLM judge：主观质量维度的评分模块（P6 第二步）。

只判规则盲区的主观维度（回答质量 / 过程合理性），按用例的 judge_aspects
声明选择性触发；规则分仍是唯一通过门槛，judge 分 advisory（不参与 pass/fail）。

防抖动三件套：
- temperature=0 + 固定版本号的 rubric prompt（当前 v2）——同回复同分数才可归因；
- 输出强制 JSON（{"score": 1-5, "reasons": [...]}），解析失败/越界重试一次；
- 再失败返回 None → 报告标"未判"，宁可不判不打假分。

v2 变更（修复 v1 的 multi-turn 材料错位 bug）：
- 材料改为按轮组织：question / reply / 当轮 tool_calls 与轮次对齐——v1 只给
  最后一轮问题却给全会话调用序列，multi-turn 用例第一轮的正当工作被判
  "与当前问题无关"（实测 multiturn_switch_consult 冤枉 1/5）；
- process rubric 补充 products-first 双路设计说明：catalog+products 并调是
  设计内的合规校验模式，不计冗余；同工具同参数高度重叠仍计冗余；
- max_tokens 1024→2048 + reasons 限 2 条——修复长 reasons 截断导致的"未判"。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

JUDGE_PROMPT_VERSION = "v2"
JUDGE_TEMPERATURE = 0

# 解析失败证据落盘（批次 2.5）："未判"必须可追溯
_FAIL_LOG_PATH = Path(__file__).resolve().parent / "runs" / "judge_failures.jsonl"

_ASPECT_RUBRICS = {
    "answer_quality": (
        "你是电商导购 Agent 的回答质量评审员。对话可能有多轮，材料会给出各轮用户问题；"
        "你只评「Agent 的最终回答」的质量，不评事实正确性（那是规则判分的事）。\n"
        "评分标准：\n"
        "1 分：答非所问，或明显冗长堆砌、结构混乱；\n"
        "2 分：沾边但重点缺失，需要用户再追问才能得到答案；\n"
        "3 分：回答了问题，但结构松散或理由单薄；\n"
        "4 分：直接回应了问题，结构清晰，理由有据；\n"
        "5 分：在 4 分基础上语言简洁专业、有恰当的引导或对比，读完即可决策。\n"
        "只输出 JSON：{\"score\": 整数1-5, \"reasons\": [\"理由1\", \"理由2\"]}"
        "（reasons 最多 2 条，每条不超过 40 字）"
    ),
    "process": (
        "你是电商导购 Agent 的工具调用过程评审员。材料按轮组织：每轮的「用户问题」"
        "与其「当轮工具调用序列」（含工具名与参数）对齐，请逐轮评估各轮调用的必要性"
        "与效率，再给整体打 1-5 分（整数）。\n"
        "评分标准：\n"
        "1 分：存在重复调用同一工具同参数、或大量与当轮问题无关的调用；\n"
        "2 分：有明显的多余调用，或漏掉了完成回答必需的关键工具；\n"
        "3 分：基本合理，但存在可省略的调用或顺序不够高效；\n"
        "4 分：每次调用都必要，顺序合理，无冗余；\n"
        "5 分：在 4 分基础上，调用规划展现出对任务的最短路径意识（最少次数覆盖全部信息需求）。\n"
        "设计说明：search_catalog（确定性过滤：品类/品牌/价格边界）与 search_products"
        "（语义检索）双路并调是设计内的合规校验模式，不计为冗余；但同一工具重复调用"
        "且参数高度重叠仍计冗余。\n"
        "只输出 JSON：{\"score\": 整数1-5, \"reasons\": [\"理由1\", \"理由2\"]}"
        "（reasons 最多 2 条，每条不超过 40 字）"
    ),
}


def _parse_judge_response(raw: str) -> dict | None:
    """解析并校验 judge 输出；不合法返回 None（由调用方决定重试/放弃）。"""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    score = data.get("score")
    # 容忍 4.0 这类整数型浮点（模型偶发输出），非整数浮点仍拒绝
    if isinstance(score, float) and score.is_integer():
        score = int(score)
    reasons = data.get("reasons")
    if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
        return None
    if not isinstance(reasons, list) or not reasons:
        reasons = []
    return {"score": score, "reasons": [str(r) for r in reasons][:2]}  # 与 rubric 的 2 条上限一致


def _log_parse_failure(aspect: str, raw: str, finish_reason: str) -> None:
    """解析最终失败时把原始输出落盘——"未判"必须留下可查证据（批次 2.5）。

    追加写到 app/evaluation/runs/judge_failures.jsonl（一行一条）。记录
    finish_reason：length = 输出被截断（该加预算），stop = 格式不合规（该改 prompt），
    两类问题的修法不同，所以必须区分记录。
    """
    try:
        with _FAIL_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "aspect": aspect,
                "rubric_version": JUDGE_PROMPT_VERSION,
                "finish_reason": finish_reason,
                "raw": raw[:4000],
            }, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001 —— 诊断日志失败不能影响评测
        logger.warning("judge 失败日志写入失败（忽略）: %s", e)


def _judge_once(client: OpenAI, model: str, system_prompt: str,
                user_content: str, call_name: str = "") -> tuple:
    """单次判分调用；返回 (解析结果或 None, 原始输出, finish_reason)。

    call_name 仅在调用方确认 client 已被 langfuse 包装时传入（看板可读性）。
    """
    kwargs = {"name": call_name} if call_name else {}
    response = client.chat.completions.create(
        model=model,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=2048,  # v1 的 1024 偶发被长 reasons 耗尽 → content 截断 → 解析失败"未判"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        **kwargs,
    )
    content = response.choices[0].message.content or ""
    finish_reason = getattr(response.choices[0], "finish_reason", "") or ""
    if finish_reason != "stop":
        return None, content, finish_reason
    return _parse_judge_response(content), content, finish_reason


def judge_aspect(
    client: OpenAI,
    model: str,
    aspect: str,
    turns: list[dict],
    call_name: str = "",
) -> dict | None:
    """对指定维度判分。turns 为按轮组织的材料：

    [{question: 该轮用户消息, reply: 该轮最终回复, tool_calls: [{name, arguments}]}, ...]

    - answer_quality：材料 = 各轮用户问题 + 最终回复（只评最终回复）；
    - process：逐轮给出「该轮问题 → 该轮调用序列」，消除 v1 材料错位。
    call_name：传入时作为 Langfuse observation 名（调用方需确认 client 已被包装）。
    失败重试一次，再失败返回 None（报告标"未判"），并把原始输出落盘留证。
    """
    rubric = _ASPECT_RUBRICS.get(aspect)
    if rubric is None or not turns:
        return None

    if aspect == "answer_quality":
        dialogue = "\n".join(
            f"第 {i + 1} 轮用户：{t['question']}" for i, t in enumerate(turns)
        )
        final = turns[-1].get("reply") or "（无回复）"
        user_content = f"【对话（按轮）】\n{dialogue}\n\n【Agent 的最终回答】\n{final}"
    else:  # process
        blocks = []
        for i, t in enumerate(turns):
            calls = "\n".join(
                f"  {j + 1}. {c['name']}({json.dumps(c.get('arguments', {}), ensure_ascii=False)})"
                for j, c in enumerate(t.get("tool_calls") or [])
            ) or "  （本轮未调用工具）"
            blocks.append(f"第 {i + 1} 轮用户：{t['question']}\n第 {i + 1} 轮工具调用：\n{calls}")
        user_content = "【按轮对齐的工具调用材料】\n" + "\n\n".join(blocks)

    result, raw, finish_reason = _judge_once(client, model, rubric, user_content, call_name)
    if result is None:
        # 格式抖动偶发：重试一次，再失败才放弃
        result, raw, finish_reason = _judge_once(
            client, model, rubric, user_content, call_name
        )
    if result is None:
        _log_parse_failure(aspect, raw, finish_reason)  # "未判"必须留下证据
    else:
        result["prompt_version"] = JUDGE_PROMPT_VERSION  # 落盘可归因：分数对应哪版 rubric
    return result
