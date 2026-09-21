"""工具结果截断视图测试（`context_budget.tool_result_view`）。

三条硬约束：
1. 不超限 → 原样返回，不做任何加工；
2. 超限 → head / tail 都在，且**总长不超过 max_chars**（引号密集导致转义膨胀时也不例外）；
3. 预算要真的用满——曾经因系数写错（`// 12`）只用到 20%，白白浪费 80% 的上下文预算。

纯函数测试，无外部依赖。
"""

import json

import pytest

from app.agent.context_budget import tool_result_view

BUDGET = 12000


def _content(chars: int, quote_heavy: bool = False) -> str:
    """造一段指定长度的内容；quote_heavy 模拟 JSON 串那种引号密集的情形。"""
    unit = '{"key": "value"}, ' if quote_heavy else "平稳的中文内容，没有特殊字符。"
    return (unit * (chars // len(unit) + 1))[:chars]


def test_within_budget_is_returned_as_is():
    content = _content(1000)
    assert tool_result_view(content, BUDGET) == content


def test_too_small_budget_raises():
    with pytest.raises(ValueError):
        tool_result_view(_content(1000), 255)


def test_truncated_view_keeps_head_and_tail_within_budget():
    content = _content(20000)
    view = tool_result_view(content, BUDGET)

    payload = json.loads(view)                      # 仍是合法 JSON（模型能正常解析）
    assert payload["truncated"] is True
    assert payload["original_chars"] == 20000
    assert payload["head"] and payload["tail"]      # 头尾都在
    assert content.startswith(payload["head"])
    assert content.endswith(payload["tail"])
    assert len(view) <= BUDGET                      # ← 硬约束


def test_budget_is_actually_used():
    """回归：曾因 `(max_chars - 200) // 12` 这个系数只用到 20% 预算。"""
    view = tool_result_view(_content(20000), BUDGET)
    assert len(view) >= BUDGET * 0.8


def test_quote_heavy_content_still_within_budget():
    """引号密集内容转义后更长——兜底回缩要保证硬上限不被突破。"""
    content = _content(30000, quote_heavy=True)
    view = tool_result_view(content, BUDGET)

    assert len(view) <= BUDGET
    assert json.loads(view)["truncated"] is True


def test_char_counters_are_self_consistent():
    """shown + omitted 恒等于 original；shown 按原文字符口径（不含 JSON 骨架与转义）。"""
    content = _content(20000)
    payload = json.loads(tool_result_view(content, BUDGET))

    assert payload["original_chars"] == 20000
    assert payload["shown_chars"] == len(payload["head"]) + len(payload["tail"])
    assert payload["shown_chars"] + payload["omitted_chars"] == payload["original_chars"]
    assert payload["omitted_chars"] > 0


def test_char_counters_stay_consistent_after_shrink():
    """引号密集会触发回缩——回缩路径最容易漏更新计数，这里专门盯它。"""
    content = _content(30000, quote_heavy=True)
    payload = json.loads(tool_result_view(content, BUDGET))

    assert payload["shown_chars"] == len(payload["head"]) + len(payload["tail"])
    assert payload["shown_chars"] + payload["omitted_chars"] == payload["original_chars"]


def test_small_budget_still_returns_valid_json():
    """预算压到下限附近也不能产出坏 JSON。"""
    view = tool_result_view(_content(5000), 256)
    assert isinstance(json.loads(view), dict)
    assert len(view) <= 256
