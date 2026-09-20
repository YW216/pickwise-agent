"""工具参数闸门测试：协议层（JSON）与契约层（schema）分层验证。

覆盖 validation.py 的两道闸门与三类处置（归一化 / 夹紧 / 拒绝），
以及 registry.execute_tool 的接线（拒绝不执行、notice 注入）。

不依赖 PG / Milvus：涉及外部服务的工具只走"校验就失败"的路径，
需要真正执行工具的用例统一用 load_skill（无外部依赖）。
"""

import pytest

from app.agent.tools.registry import TOOL_DEFINITIONS, execute_tool
from app.agent.tools.validation import (
    ArgumentError,
    parse_tool_arguments,
    validate_arguments,
)


def _schema(tool_name: str) -> dict:
    """按名取工具 schema：直接复用生产定义，避免测试里再抄一份规则。"""
    for definition in TOOL_DEFINITIONS:
        if definition["function"]["name"] == tool_name:
            return definition["function"]["parameters"]
    raise AssertionError(f"schema not found: {tool_name}")


# ---------- 协议层：arguments 字符串解析 ----------

@pytest.mark.parametrize(
    ("raw", "expected_ok"),
    [
        ('{"query": "耳机"}', True),
        ("{}", True),
        ("", False),
        (None, False),
        ("   ", False),
        ('{"query": "通勤耳', False),   # 截断——思考与正文共享输出预算的典型形态
        ("[1, 2]", False),              # JSON 合法但不是对象
        ('"just a string"', False),
        ("{not json}", False),
    ],
)
def test_parse_tool_arguments(raw, expected_ok):
    parsed, error = parse_tool_arguments(raw)
    if expected_ok:
        assert error is None
        assert isinstance(parsed, dict)
    else:
        assert parsed is None
        assert error  # 回喂文案非空，模型据此重发


# ---------- 契约层：无损归一化（静默修） ----------

def test_number_string_is_normalized_silently():
    args, notes = validate_arguments(
        _schema("search_products"), {"query": "耳机", "limit": "10"},
    )
    assert args["limit"] == 10
    assert isinstance(args["limit"], int)
    assert notes == []  # 字符串 "10" 无歧义，不打扰模型


def test_optional_null_is_dropped():
    args, notes = validate_arguments(
        _schema("search_products"), {"query": "耳机", "category": None},
    )
    assert "category" not in args
    assert notes == []


# ---------- 契约层：边界夹紧（修 + 告知） ----------

def test_limit_above_max_is_clamped_with_notice():
    args, notes = validate_arguments(
        _schema("search_products"), {"query": "耳机", "limit": 1000},
    )
    assert args["limit"] == 20
    assert any("上限" in note for note in notes)


def test_limit_below_min_is_clamped_with_notice():
    args, notes = validate_arguments(
        _schema("search_products"), {"query": "耳机", "limit": 0},
    )
    assert args["limit"] == 1
    assert notes


def test_array_above_max_items_is_truncated_not_rejected():
    args, notes = validate_arguments(
        _schema("compare_products"),
        {"product_ids": ["LP-01", "LP-02", "LP-03", "LP-04", "LP-05"]},
    )
    assert args["product_ids"] == ["LP-01", "LP-02", "LP-03", "LP-04"]
    assert notes


def test_unknown_argument_is_filtered_with_available_list():
    # agent.log 里的真实形态：模型给 search_catalog 塞了 schema 中不存在的 conditions
    args, notes = validate_arguments(
        _schema("search_catalog"),
        {"category": "耳机", "conditions": ["入耳式"]},
    )
    assert "conditions" not in args
    assert args["category"] == "耳机"
    assert any("conditions" in note and "可用参数" in note for note in notes)


# ---------- 契约层：语义拒绝（回喂改值） ----------

def test_enum_violation_is_rejected():
    with pytest.raises(ArgumentError) as excinfo:
        validate_arguments(
            _schema("search_products"),
            {"query": "笔记本", "category": "笔记本电脑"},
        )
    assert "category" in str(excinfo.value)


def test_missing_required_is_rejected():
    with pytest.raises(ArgumentError):
        validate_arguments(_schema("search_products"), {"limit": 3})


def test_array_below_min_items_is_rejected():
    with pytest.raises(ArgumentError):
        validate_arguments(
            _schema("compare_products"), {"product_ids": ["LP-01"]},
        )


def test_wrong_type_is_rejected():
    with pytest.raises(ArgumentError):
        validate_arguments(
            _schema("search_products"), {"query": "耳机", "limit": [1]},
        )


def test_float_for_integer_is_rejected():
    # 不取整：取整会静默改变模型的本意
    with pytest.raises(ArgumentError):
        validate_arguments(
            _schema("search_products"), {"query": "耳机", "limit": 1.5},
        )


def test_blank_string_is_rejected():
    with pytest.raises(ArgumentError):
        validate_arguments(_schema("search_products"), {"query": "   "})


# ---------- 执行器接线：拒绝不执行、notice 注入 ----------

def test_execute_tool_rejects_unknown_name():
    result = execute_tool("no_such_tool", {})
    assert result["success"] is False
    assert "未知工具" in result["error"]


def test_execute_tool_rejects_non_dict_arguments():
    result = execute_tool("load_skill", ["skill_name"])
    assert result["success"] is False


def test_execute_tool_returns_contract_error_without_executing():
    # 空 query 属契约层拒绝：不应走到 Milvus，文案里也不能出现"检索不可用"
    result = execute_tool("search_products", {"query": ""})
    assert result["success"] is False
    assert "不能为空" in result["error"]


def test_execute_tool_appends_notice_on_clamp():
    result = execute_tool("search_catalog", {"limit": 1000})
    assert "notice" in result
    assert "上限" in result["notice"]


def test_execute_tool_adds_no_notice_when_arguments_untouched():
    result = execute_tool("load_skill", {"skill_name": "product-recommend"})
    assert "notice" not in result
