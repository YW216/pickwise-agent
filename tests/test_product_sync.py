"""product_sync 纯逻辑单元测试：diff 五分类 / 契约解析 / 模板稳定性。

plan_sync / parse_contract / build_card_text 是纯函数——不依赖真实
PG / Milvus / embedding API。真库的 upsert / 删除 / 失败恢复验证
属集成测试（第 4 步 test_product_kb_incremental.py）。

对应方案验收项（develop_docs/rag增量更新.md 第八节 11.1）：
- 无变化任务绝不产生 embed 计划（added + text_changed 为空）
- specs 键序扰动 hash 不变；价格不进 text
- 空快照在 plan 层表现为"全部删除"，防误删保护在 sync 层
"""

from datetime import datetime, timedelta
import hashlib

from app.services.product_sync import (
    build_card_text,
    build_contract,
    card_hash,
    parse_contract,
    plan_sync,
)

BASE = datetime(2026, 10, 10, 12, 0, 0)


def row(pid="PH-01", **over):
    """构造一条快照行（业务字段 + 同步状态列），over 覆盖任意字段。"""
    r = {
        "product_id": pid,
        "name": "星海 凌霄 Phone 5",
        "brand": "星海",
        "category": "手机",
        "price": 3999,
        "specs": {"芯片": "X3", "内存": "12G"},
        "introduction": ["旗舰影像，全能体验。"],
        "updated_at": BASE,
        "card_hash": None,
        "indexed_at": None,
    }
    r.update(over)
    return r


def synced(r, text_hash=None):
    """置为"已同步"状态：card_hash=当前文本指纹，indexed_at 晚于 updated_at。"""
    r["card_hash"] = text_hash or card_hash(build_card_text(r))
    r["indexed_at"] = r["updated_at"] + timedelta(hours=1)
    return r


def later(r, hours=2):
    """模拟业务修改后触发器刷新 updated_at。"""
    r["updated_at"] = BASE + timedelta(hours=hours)
    return r


# ---------- diff 五分类 ----------


def test_first_run_all_added():
    """首次扫描：状态全 NULL → 全部识别为新增。"""
    plan = plan_sync({"PH-01": row(), "PH-02": row("PH-02")}, set())
    assert plan.added == ["PH-01", "PH-02"]
    assert plan.text_changed == [] and plan.row_changed == []
    assert plan.unchanged == [] and plan.deleted == []


def test_unchanged_no_embed_plan():
    """无变化重跑：全部 unchanged，embed 候选（added+text_changed）为空。"""
    r = synced(row())
    plan = plan_sync({"PH-01": r}, {"PH-01"})
    assert plan.unchanged == ["PH-01"]
    assert plan.added == [] and plan.text_changed == []  # → 零 embedding 调用


def test_text_changed_only_that_row():
    """只改一款商品的文本 → 仅该商品进 text_changed。"""
    r1, r2 = synced(row("PH-01")), synced(row("PH-02"))
    r1["name"] = "星海 凌霄 Phone 5S"
    later(r1)
    plan = plan_sync({"PH-01": r1, "PH-02": r2}, {"PH-01", "PH-02"})
    assert plan.text_changed == ["PH-01"]
    assert plan.unchanged == ["PH-02"]


def test_price_change_is_row_changed_not_text_changed():
    """只改价格（不进 text）→ row_changed，而非重算向量的 text_changed。"""
    r = synced(row())
    r["price"] = 3899
    later(r)
    plan = plan_sync({"PH-01": r}, {"PH-01"})
    assert plan.row_changed == ["PH-01"]
    assert plan.text_changed == []  # 核心：改价零 embedding


def test_delete_detected_from_milvus_minus_pg():
    """Milvus 有、PG 无 → 删除集合恰为该 id。"""
    plan = plan_sync({"PH-01": synced(row())}, {"PH-01", "PH-99"})
    assert plan.deleted == ["PH-99"]


def test_empty_snapshot_yields_all_deleted():
    """空快照在 plan 层 = 全部删除；防误删中止保护在 sync 层（本测只锁 plan 语义）。"""
    plan = plan_sync({}, {"PH-01", "PH-02"})
    assert plan.deleted == ["PH-01", "PH-02"]


# ---------- 模板 / 指纹稳定性 ----------


def test_specs_key_order_does_not_change_hash():
    """specs 字典键序不同 → 渲染文本与 hash 均稳定（防御数据源键序漂移）。"""
    a = row(specs={"内存": "12G", "芯片": "X3"})
    b = row(specs={"芯片": "X3", "内存": "12G"})
    assert build_card_text(a) == build_card_text(b)
    assert card_hash(build_card_text(a)) == card_hash(build_card_text(b))


def test_price_not_in_card_text():
    """v2 模板：价格不出现在卡片文本中（改价不触发向量重算的前提）。"""
    text = build_card_text(row(price=12345))
    assert "售价" not in text
    assert "12345" not in text


def test_hash_matches_content():
    """指纹是对文本的 SHA-256，文本变则指纹必变。"""
    t1 = build_card_text(row())
    t2 = build_card_text(row(name="星海 凌霄 Phone 5S"))
    assert card_hash(t1) != card_hash(t2)
    assert card_hash(t1) == hashlib.sha256(t1.encode("utf-8")).hexdigest()


# ---------- 契约解析 ----------


def test_contract_roundtrip():
    """build → parse 往返一致。"""
    c = build_contract("BAAI/bge-m3")
    assert parse_contract(c) == {
        "embedding_model": "BAAI/bge-m3",
        "template": "product_card_template_v2",
    }


def test_parse_contract_garbage_is_empty():
    """垃圾/空串解析为空 dict（调用方按"契约缺失"处理，拒绝增量）。"""
    assert parse_contract("") == {}
    assert parse_contract("不是键值对") == {}
