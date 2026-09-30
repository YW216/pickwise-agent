"""压缩策略的纯函数与状态契约测试（破坏性删除版：无 cursor）。

契约对应（8-压缩重新设计.md 第二版 + v2.1 商品记忆独立存储）：
- should_compact：token 阈值 + reserve 预算（D8）
- find_cut_point：只在已完成 user turn 边界切分，切点 = 被删前缀长度（C2）
- compact：只摘要本段增量（delta），商品记忆已独立至 product_tracker（D14）；不修改输入列表
- 摘要校验：六段齐全且非空，失败重试一次（D12：宁可不压）
- product_tracker：工具 JSON 是唯一可信的名称/价格来源；纯函数；随会话持久化
- storage：messages + summary + products 落盘；旧版本会话文件作废（D11）
"""

import json
from types import SimpleNamespace

import pytest

from app.agent.compaction import (
    compact,
    find_cut_point,
    should_compact,
)
from app.agent.product_tracker import format_products_block, merge_products
from app.agent.storage import SESSION_VERSION, load_session, save_session


STRUCTURED_SUMMARY = """## 用户目标
买一台适合编程的笔记本

## 约束与偏好
- 预算有限，重视续航

## 进展
### 已完成
- [x] 已查询候选
### 进行中
- [ ] 继续确认
### 受阻
- （无）

## 关键决策
- （无）

## 下一步
1. 继续比较候选

## 关键上下文
- 保留商品 ID 和价格
"""


class FakeClient:
    def __init__(self, *contents: str):
        self.contents = list(contents)
        self.calls = []

    @property
    def chat(self):
        return SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.contents[index]))]
        )


def test_should_compact_uses_reserve_budget():
    assert should_compact(91, 100, 10)
    assert not should_compact(89, 100, 10)
    assert not should_compact(1000, 100, 10, enabled=False)


def test_find_cut_point_keeps_current_turn_and_never_cuts_tool():
    messages = [
        {"role": "user", "content": "第一轮"},
        {"role": "assistant", "content": "第一轮答复"},
        {"role": "user", "content": "第二轮，查 LP-01"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "工具结果"},
        {"role": "assistant", "content": "第二轮答复"},
        {"role": "user", "content": "第三轮，当前问题"},
    ]

    cut = find_cut_point(messages, keep_recent_tokens=1, end=6)

    # 切点 = 保留区第一条（第二轮 user），第一、二轮整体进入待删前缀
    assert cut == 2
    assert messages[cut]["role"] == "user"
    assert messages[cut]["role"] != "tool"


def test_find_cut_point_returns_zero_without_completed_turn():
    """没有可切的完整 turn（首轮未完成 / 只有当前轮）→ 返回 0，宁可不压。"""
    messages = [
        {"role": "user", "content": "当前问题"},
        {"role": "assistant", "content": "答复"},
    ]
    assert find_cut_point(messages, keep_recent_tokens=1, end=2) == 0

    # end=1：范围 (0,1) 内无 user 起点 → 无可删前缀
    messages_two_turns = [
        {"role": "user", "content": "第一轮"},
        {"role": "user", "content": "第二轮"},
    ]
    assert find_cut_point(messages_two_turns, keep_recent_tokens=1, end=1) == 0

    # 累积到列表头部仍不够 keep_recent_tokens → 0（兜底：不全删，D12）
    assert find_cut_point(messages_two_turns, keep_recent_tokens=10**9, end=2) == 0


def test_compact_sends_only_delta():
    """摘要只接收待删前缀（delta），不含当前轮；压缩器不修改输入列表。"""
    messages = [
        {"role": "user", "content": "比较 LP-01"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1"}]},
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": '{"success":true,"data":{"product":{"product_id":"LP-01","name":"星海凌霄","price":5999}}}',
        },
        {"role": "assistant", "content": "LP-01 ¥5999"},
        {"role": "user", "content": "旧消息"},
        {"role": "assistant", "content": "旧答复"},
        {"role": "user", "content": "当前问题"},
    ]
    client = FakeClient(STRUCTURED_SUMMARY)
    snapshot = [dict(m) for m in messages]

    cut, summary = compact(
        messages=messages,
        summary=None,
        client=client,
        model="test-model",
        keep_recent_tokens=1,
        end=6,
    )

    assert cut == 4  # messages[:4] 即待删前缀
    assert "比较 LP-01" in client.calls[0]["messages"][1]["content"]
    assert "当前问题" not in client.calls[0]["messages"][1]["content"]
    assert summary == STRUCTURED_SUMMARY.strip()  # 纯六段叙述，商品块不再由压缩器生成
    assert client.calls[0]["max_tokens"] == 4096
    # 压缩器是纯函数：不修改传入列表（删除由调用方事务性执行）
    assert messages == snapshot


def test_compact_rejects_unstructured_summary():
    messages = [
        {"role": "user", "content": "第一轮"},
        {"role": "assistant", "content": "答复"},
        {"role": "user", "content": "当前问题"},
    ]
    client = FakeClient("自由文本")

    with pytest.raises(ValueError, match="固定 section"):
        compact(
            messages=messages,
            summary=None,
            client=client,
            model="test-model",
            keep_recent_tokens=1,
            end=3,
        )
    # 两次都失败才放弃：调用恰好 2 次
    assert len(client.calls) == 2


def test_compact_retries_once_and_tolerates_trailing_whitespace():
    """第一次格式抖动、第二次正常 → 重试成功；标题尾空格被规范化容忍。"""
    messages = [
        {"role": "user", "content": "第一轮"},
        {"role": "assistant", "content": "答复"},
        {"role": "user", "content": "当前问题"},
    ]
    spaced = "\n".join(
        line + "  " if line.startswith("## ") else line
        for line in STRUCTURED_SUMMARY.splitlines()
    )
    client = FakeClient("自由文本", spaced)

    cut, summary = compact(
        messages=messages,
        summary=None,
        client=client,
        model="test-model",
        keep_recent_tokens=1,
        end=3,
    )

    assert len(client.calls) == 2
    assert cut == 2
    assert summary == STRUCTURED_SUMMARY.strip()


def test_compact_accepts_summary_at_realistic_length():
    """回归（2026-09-29 真调踩坑）：摘要上限原为 2048 token。

    实测第一次压缩生成的摘要就有 1701 字（约 1560 token），已逼近旧上限；
    第二次压缩必然超限抛错 → 摘要只增不减 → 此后永久压不动 → 上下文涨到溢出。
    上限提到 4096 token / 3000 字符后，这个规模应能正常通过。
    """
    filler = "候选机型的参数与结论记录。" * 120
    at_limit = STRUCTURED_SUMMARY.replace("- 保留商品 ID 和价格", filler)
    assert 1500 < len(at_limit) < 3000, "样本长度应贴近实测规模（1701 字）"

    messages = [
        {"role": "user", "content": "第一轮"},
        {"role": "assistant", "content": "答复"},
        {"role": "user", "content": "当前问题"},
    ]
    cut, summary = compact(
        messages=messages,
        summary=None,
        client=FakeClient(at_limit),
        model="test-model",
        keep_recent_tokens=1,
        end=3,
    )

    assert cut == 2
    assert summary == at_limit.strip()


def test_compact_condenses_overlong_summary():
    """补救式自我压缩：增量摘要越限时，应调 condense 压回并提交精简版。"""
    overlong = STRUCTURED_SUMMARY.replace("- 保留商品 ID 和价格", "填充" * 1600)
    assert len(overlong) > 3000, "样本需超过 summary_max_chars"

    messages = [
        {"role": "user", "content": "第一轮"},
        {"role": "assistant", "content": "答复"},
        {"role": "user", "content": "当前问题"},
    ]
    # 第 1 次调用返回越限摘要（触发 condense），第 2 次返回精简版
    client = FakeClient(overlong, STRUCTURED_SUMMARY)

    cut, summary = compact(
        messages=messages,
        summary=None,
        client=client,
        model="test-model",
        keep_recent_tokens=1,
        end=3,
    )

    assert cut == 2
    assert summary == STRUCTURED_SUMMARY.strip()   # 保存的是精简版，不是越限版
    assert len(client.calls) == 2                  # 增量生成 1 次 + condense 1 次


def test_compact_fails_when_condense_still_overlong():
    """condense 之后仍越限 → 判失败（保持零副作用，绝不把越限摘要写进历史）。"""
    overlong = STRUCTURED_SUMMARY.replace("- 保留商品 ID 和价格", "填充" * 1600)

    messages = [
        {"role": "user", "content": "第一轮"},
        {"role": "assistant", "content": "答复"},
        {"role": "user", "content": "当前问题"},
    ]
    client = FakeClient(overlong, overlong)        # 两次都越限

    with pytest.raises(ValueError, match="长度上限"):
        compact(
            messages=messages,
            summary=None,
            client=client,
            model="test-model",
            keep_recent_tokens=1,
            end=3,
        )

    assert len(client.calls) == 2


def test_merge_products_extracts_from_tool_json_and_inherits_store():
    """工具 JSON 是唯一可信的名称/价格来源；已有 store 原样继承（跨压缩累积）。"""
    store = {"LP-01": {"product_id": "LP-01", "name": "星海凌霄", "price_at_mention": 5999}}
    messages = [
        {"role": "user", "content": "那 LP-03 多少钱"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "get_detail", "arguments": '{"product_id": "LP-05"}'}},
        ]},
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": '{"success":true,"data":{"product":{"product_id":"LP-03","name":"曜石长风","price":3299}}}',
        },
    ]

    merged = merge_products(store, messages)

    assert merged["LP-01"] == store["LP-01"]  # 旧店继承
    assert merged["LP-03"] == {"product_id": "LP-03", "name": "曜石长风", "price_at_mention": 3299}
    assert merged["LP-05"] == {"product_id": "LP-05"}  # tool_calls 参数里只登记壳子
    # 纯函数：不改入参
    assert store == {"LP-01": {"product_id": "LP-01", "name": "星海凌霄", "price_at_mention": 5999}}


def test_merge_products_matches_id_adjacent_to_chinese():
    """中文紧贴的商品 ID 必须能登记（单词边界在中文旁不成立，已改否定环视）。"""
    messages = [
        {"role": "assistant", "content": "推荐LP-01这款；价格LP-03为5999"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "get_detail", "arguments": '{"product_id":"LP-05"}'}},
        ]},
    ]

    merged = merge_products({}, messages)

    assert set(merged) == {"LP-01", "LP-03", "LP-05"}
    assert all(merged[pid] == {"product_id": pid} for pid in merged)  # 文本来源只登记壳子


def test_merge_products_ignores_fused_tokens():
    """XLP-01 / LP-011 不是独立编号，不得登记。"""
    messages = [{"role": "assistant", "content": "XLP-01、LP-011 无效，LP-05 有效"}]

    assert set(merge_products({}, messages)) == {"LP-05"}


def test_format_products_block():
    products = {"LP-01": {"product_id": "LP-01", "name": "星海凌霄", "price_at_mention": 5999}}

    block = format_products_block(products)
    assert block.startswith("<mentioned-products>") and block.endswith("</mentioned-products>")
    assert "星海凌霄" in block and "5999" in block
    assert format_products_block({}) == ""  # 空店 → 空串，注入段整体省略


def test_session_round_trip_with_products(tmp_path):
    path = tmp_path / "session.json"
    messages = [
        {"role": "user", "content": "测试"},
        {"role": "assistant", "content": "答复"},
    ]
    products = {"LP-01": {"product_id": "LP-01", "name": "星海凌霄", "price_at_mention": 5999}}

    save_session(str(path), messages, STRUCTURED_SUMMARY, products=products)
    loaded = load_session(str(path))

    assert loaded["messages"] == messages
    assert loaded["summary"] == STRUCTURED_SUMMARY
    assert loaded["products"] == products


def test_session_rejects_legacy_and_invalid_files(tmp_path):
    """cursor 时代（v2/v3）会话文件作废；首条非 user 的状态不合法。"""
    for version, payload in (
        (2, {"summary": STRUCTURED_SUMMARY, "cursor": 4,
             "messages": [{"role": "assistant", "content": "旧格式残留"}]}),
        (3, {"summary": STRUCTURED_SUMMARY,
             "messages": [{"role": "user", "content": "v3 旧格式"}]}),
    ):
        path = tmp_path / f"legacy_v{version}.json"
        path.write_text(json.dumps({"version": version, **payload}, ensure_ascii=False), encoding="utf-8")
        assert load_session(str(path)) is None

    invalid = {
        "version": SESSION_VERSION,
        "summary": STRUCTURED_SUMMARY,
        "messages": [{"role": "assistant", "content": "窗口起点不是 user turn"}],
    }
    path_bad = tmp_path / "invalid.json"
    path_bad.write_text(json.dumps(invalid, ensure_ascii=False), encoding="utf-8")
    assert load_session(str(path_bad)) is None

    with pytest.raises(ValueError, match="不合法"):
        save_session(str(tmp_path / "bad.json"), [{"role": "assistant", "content": "x"}], None)
