"""重复调用熔断测试（harness H5）。

规则：同一签名（工具名 + **归一化后**参数）在本轮内最多执行 MAX_SAME_CALLS 次，
第 3 次起拦下并回喂；拦下的那次不计入计数，因此到上限后每次都被拦。

不依赖 PG / Milvus：
- 执行路径统一用 `load_skill`（无外部依赖）；
- 需要"直接触发熔断"的用例预置 `seen` 计数，让工具根本不被执行。
"""

import json

from app.agent.tools.registry import (
    MAX_SAME_CALLS,
    TOOL_DEFINITIONS,
    _signature,
    execute_tool,
)
from app.agent.tools.validation import validate_arguments


def _schema(tool_name: str) -> dict:
    """按名取工具 schema（复用生产定义，不另抄一份规则）。"""
    for definition in TOOL_DEFINITIONS:
        if definition["function"]["name"] == tool_name:
            return definition["function"]["parameters"]
    raise AssertionError(f"schema not found: {tool_name}")


def _is_blocked(result: dict) -> bool:
    """熔断的判定：错误文案来自熔断而非工具本身。"""
    return result.get("success") is False and "不要重复调用" in (result.get("error") or "")


# ---------- 签名规则 ----------

def test_signature_is_key_order_independent():
    a = _signature("search_products", {"query": "耳机", "limit": 5})
    b = _signature("search_products", {"limit": 5, "query": "耳机"})
    assert a == b


def test_signature_uses_normalized_arguments():
    """签名建在归一化之后："10" 与 10 必须是同一个签名，否则换个写法就绕过了熔断。"""
    schema = _schema("search_products")
    raw, _ = validate_arguments(schema, {"query": "耳机", "limit": "10"})
    cleaned, _ = validate_arguments(schema, {"query": "耳机", "limit": 10})
    assert _signature("search_products", raw) == _signature("search_products", cleaned)


# ---------- 阈值与计数 ----------

def test_third_same_call_is_blocked_with_actionable_error():
    seen: dict[str, int] = {}
    args = {"skill_name": "product-recommend"}

    for _ in range(MAX_SAME_CALLS):
        assert not _is_blocked(execute_tool("load_skill", args, seen))

    blocked = execute_tool("load_skill", args, seen)
    assert _is_blocked(blocked)
    # 文案要能独立读懂：工具名 + 参数、次数、以及三条出路
    assert "load_skill" in blocked["error"]
    assert "product-recommend" in blocked["error"]
    assert "两次" in blocked["error"]
    assert "前面的工具消息" in blocked["error"]
    assert "换查询条件" in blocked["error"]


def test_blocked_call_does_not_increment_count():
    """拦下的那次不计入，所以到上限后每一次同签名调用都被拦。"""
    seen: dict[str, int] = {}
    args = {"skill_name": "product-recommend"}
    for _ in range(MAX_SAME_CALLS):
        execute_tool("load_skill", args, seen)

    assert _is_blocked(execute_tool("load_skill", args, seen))
    assert _is_blocked(execute_tool("load_skill", args, seen))   # 第 4 次仍被拦
    assert _is_blocked(execute_tool("load_skill", args, seen))   # 第 5 次仍被拦


def test_different_arguments_are_not_blocked():
    """同工具不同参数是正常探索（逐个查候选），零误伤。"""
    seen: dict[str, int] = {}
    for _ in range(MAX_SAME_CALLS):
        execute_tool("load_skill", {"skill_name": "product-recommend"}, seen)

    other = execute_tool("load_skill", {"skill_name": "process-return"}, seen)
    assert not _is_blocked(other)


def test_different_tool_is_not_blocked():
    seen: dict[str, int] = {}
    for _ in range(MAX_SAME_CALLS):
        execute_tool("load_skill", {"skill_name": "product-recommend"}, seen)

    other = execute_tool("recall_user_memory", {}, seen)
    assert not _is_blocked(other)


def test_fresh_seen_resets_counting():
    """跨轮不误伤：新一轮用新的 seen（生命周期 = 一次 handle）。"""
    args = {"skill_name": "product-recommend"}
    first_round: dict[str, int] = {}
    for _ in range(MAX_SAME_CALLS):
        execute_tool("load_skill", args, first_round)
    assert _is_blocked(execute_tool("load_skill", args, first_round))

    second_round: dict[str, int] = {}
    assert not _is_blocked(execute_tool("load_skill", args, second_round))


def test_no_seen_means_no_breaker():
    """不传 seen 就完全不熔断（程序侧调用与单测不受影响）。"""
    args = {"skill_name": "product-recommend"}
    for _ in range(MAX_SAME_CALLS * 3):
        assert execute_tool("load_skill", args) is not None
    assert not _is_blocked(execute_tool("load_skill", args))


# ---------- 信封形状与文案细节 ----------

def test_blocked_envelope_has_no_extra_marker_field():
    """熔断信封仍是固定三字段，不加"这是熔断"的标记——观测走 print 旁路。"""
    seen: dict[str, int] = {}
    sig = _signature("load_skill", {"skill_name": "x"})
    seen[sig] = MAX_SAME_CALLS

    result = execute_tool("load_skill", {"skill_name": "x"}, seen)
    assert set(result) == {"success", "error", "data"}
    assert result["data"] == {}


def test_long_signature_is_truncated_in_message():
    """超长参数摘要要截断，避免一次拦截反而灌进一大段文本。"""
    long_query = "轻薄的" * 60                     # 远超 120 字符
    args = {"query": long_query}
    seen: dict[str, int] = {}
    sig = _signature("search_products", args)
    seen[sig] = MAX_SAME_CALLS

    result = execute_tool("search_products", args, seen)
    assert _is_blocked(result)
    assert "..." in result["error"]
    assert long_query not in result["error"]       # 原始长参数没有整段出现
