"""product_kb 增量同步 Milvus 集成测试（临时 lite 库，绝不碰真 product_kb）。

隔离策略（develop_docs/rag增量更新.md 11.2）：
- Milvus：tmp_path 下的 milvus-lite 文件 + 与生产同构 schema（embedding
  维度缩为 8，索引/分词/BM25 Function 与生产完全一致）
- PG：monkeypatch catalog_repo 的 load_sync_snapshot / mark_products_indexed
  （sync() 函数内 import，运行时解析模块属性，patch 生效）
- embedding：确定性 FakeEmbedder（文本 hash → 固定向量），验证的是同步
  行为与可检索性，不是向量质量

用例函数接收 env（不使用 fixture 参数），便于 verify_product_kb_standalone.py
对 standalone 部署复用同一套断言。

覆盖：同 id upsert 去重 / BM25 随 text 刷新 / 标量过滤按新值 / 删除不可召回 /
embedding 故障降级（BM25 先行）→ 恢复后下轮补向量补状态。
"""

import hashlib
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from pymilvus import DataType, Function, FunctionType, MilvusClient

from app.config.settings import settings
from app.services.product_sync import build_card_text, build_contract, card_hash, sync

DIM = 8
BASE = datetime(2026, 10, 10, 12, 0, 0)


# ---------- 测试替身 ----------


class FakeEmbedder:
    """确定性假 embedder：文本 hash 前 DIM 字节 → 归一化向量；记录调用次数。"""

    def __init__(self, dim: int = DIM, fail_first: bool = False):
        self.model = "fake-embed-model"
        self.dim = dim
        self.calls = 0
        self.fail_first = fail_first
        self._failed = False

    def encode(self, texts):
        self.calls += len(texts)
        if self.fail_first and not self._failed:
            self._failed = True
            raise RuntimeError("模拟 embedding 服务故障")
        out = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            out.append([b / 255.0 for b in digest[: self.dim]])
        return out

    def encode_one(self, text):
        return self.encode([text])[0]


def make_row(pid="P1", name="星海 测试机", price=100, category="手机", **state):
    """构造快照行（业务字段 + 同步状态列）。"""
    r = {
        "product_id": pid, "name": name, "brand": "星海", "category": category,
        "price": price, "specs": {"内存": "8G"},
        "introduction": [f"{name}，主打耐用。"],
        "updated_at": BASE, "card_hash": None, "indexed_at": None,
    }
    r.update(state)
    return r


def modify_business(r: dict, **kv):
    """模拟业务修改：字段更新 + 触发器刷 updated_at（测试里手动后移）。

    name 变化时同步重写 introduction（它与 name 同源生成），保证
    "改 text"用例里旧词真正离开检索面。
    """
    r.update(kv)
    if "name" in kv:
        r["introduction"] = [f"{kv['name']}，主打耐用。"]
    r["updated_at"] = r["updated_at"] + timedelta(hours=1)
    return r


# ---------- 测试环境 ----------


def build_collection(client: MilvusClient) -> None:
    """与生产同构的 schema（维度缩为 8）。collection 名取 settings 当前值。"""
    collection = settings.product_collection
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("product_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=65535,
                     enable_analyzer=True, analyzer_params={"tokenizer": "jieba"})
    schema.add_field("price", DataType.INT64)
    schema.add_field("category", DataType.VARCHAR, max_length=32)
    schema.add_field("sparse_bm25", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                                 input_field_names=["text"], output_field_names="sparse_bm25"))
    ip = client.prepare_index_params()
    ip.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
    ip.add_index(field_name="sparse_bm25", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(
        collection_name=collection, schema=schema, index_params=ip,
        properties={"description": build_contract("fake-embed-model")},
    )


def build_env(client: MilvusClient, state: dict) -> SimpleNamespace:
    """组装 env：client / 可变快照 state / FakeEmbedder / 状态提交记录。

    state 直接被 sync 的 monkeypatch 读取——测试用例改 state 后调 sync()
    即模拟"PG 变了，跑一轮增量"。返回值带 restore()，用于恢复被替换的
    catalog_repo 函数（fixture teardown / verify 脚本收尾调用）。
    """
    embedder = FakeEmbedder()
    marks: list[dict] = []

    def fake_load(database_url):
        return {pid: dict(r) for pid, r in state.items()}

    def fake_mark(database_url, hashes, expected_updated_at=None):
        # 真实乐观锁 SQL 的行为由真库验证覆盖；这里记录提交内容供断言
        marks.append(dict(hashes))
        return len(hashes), []

    import app.db.catalog_repo as repo
    original = (repo.load_sync_snapshot, repo.mark_products_indexed)
    repo.load_sync_snapshot = fake_load
    repo.mark_products_indexed = fake_mark

    def restore():
        repo.load_sync_snapshot, repo.mark_products_indexed = original

    return SimpleNamespace(client=client, state=state, embedder=embedder,
                           marks=marks, load=fake_load, mark=fake_mark, restore=restore)


def run_sync(env):
    """跑一轮增量同步，flush 强制落盘并容忍最终一致可见延迟。"""
    stats = sync(embedder=env.embedder, client=env.client)
    env.client.flush(settings.product_collection)
    time.sleep(0.3)
    return stats


def bm25_search(client, query: str) -> list[str]:
    """BM25 单路检索（词面验证，不掺 dense 假向量）。"""
    hits = client.search(
        collection_name=settings.product_collection, data=[query],
        anns_field="sparse_bm25", search_params={"metric_type": "BM25"}, limit=10,
        output_fields=["product_id"],
    )[0]
    return [h["entity"]["product_id"] for h in hits]


def all_ids(client) -> list[str]:
    rows = client.query(collection_name=settings.product_collection, filter="",
                        output_fields=["product_id"], limit=100)
    return sorted(r["product_id"] for r in rows)


def get_row(client, pid: str, fields=("text", "price", "category", "embedding")):
    got = client.query(collection_name=settings.product_collection,
                       filter=f'product_id == "{pid}"', output_fields=list(fields), limit=1)
    return got[0] if got else None


# ---------- 用例（pytest 与 standalone verify 共用） ----------


def case_upsert_same_id_no_duplicate(env):
    """同 id 二次 upsert：不产生重复实体，text/标量为新值。"""
    env.state["P1"] = make_row("P1", price=100)
    run_sync(env)
    assert all_ids(env.client) == ["P1"]

    modify_business(env.state["P1"], name="星海 测试机 Pro", price=200)
    run_sync(env)  # name 变 → text_changed → 重算 upsert
    assert all_ids(env.client) == ["P1"]
    row = get_row(env.client, "P1")
    assert "测试机 Pro" in row["text"]
    assert row["price"] == 200


def case_bm25_refreshed_on_text_change(env):
    """text 变更后 BM25 新词可召回、旧词不再命中（sparse 由服务端 Function 重算）。"""
    env.state["P1"] = make_row("P1", name="石墨烯电池长续航机")
    run_sync(env)
    assert "P1" in bm25_search(env.client, "石墨烯电池")

    modify_business(env.state["P1"], name="太阳能充电旗舰机")
    run_sync(env)
    hits = bm25_search(env.client, "石墨烯电池")
    assert "P1" not in hits
    assert "P1" in bm25_search(env.client, "太阳能充电")


def case_scalar_filter_uses_new_value(env):
    """改价后标量过滤按新值生效（expr 与存储值一致）。"""
    env.state["P1"] = make_row("P1", price=100)
    run_sync(env)

    def hit(upper):
        rows = env.client.query(collection_name=settings.product_collection,
                                filter=f"price <= {upper}",
                                output_fields=["product_id"], limit=10)
        return [r["product_id"] for r in rows]

    assert "P1" in hit(200)
    modify_business(env.state["P1"], price=500)
    run_sync(env)  # 仅价格变 → row_changed 复用向量，标量更新
    assert "P1" not in hit(200)
    assert "P1" in hit(600)


def case_delete_not_retrievable(env):
    """PG 删除行 → 同步后该 id 从索引消失。"""
    env.state["P1"] = make_row("P1")
    env.state["P2"] = make_row("P2", name="苍岭 备用机")
    run_sync(env)
    assert all_ids(env.client) == ["P1", "P2"]

    env.state.pop("P1")
    run_sync(env)
    assert all_ids(env.client) == ["P2"]
    assert get_row(env.client, "P1") is None


def case_degrade_then_recover(env):
    """embedding 故障：填零只走 BM25 且不提交状态；恢复后下轮补向量补状态。"""
    env.embedder = FakeEmbedder(fail_first=True)
    env.state["P1"] = make_row("P1", name="降级恢复测试机")
    stats = run_sync(env)  # 故障轮：不应抛异常
    assert stats.degraded == ["P1"]
    assert env.marks == []  # 状态未提交 → 下轮必然重做

    row = get_row(env.client, "P1")
    assert row is not None  # BM25 路已上线
    assert "P1" in bm25_search(env.client, "降级恢复测试机")
    assert sum(row["embedding"]) == 0.0  # 填零向量

    stats2 = run_sync(env)  # 恢复轮：embedder 不再抛 → 补算向量 + 补状态
    assert stats2.degraded == []
    assert len(env.marks) == 1 and "P1" in env.marks[0]
    row2 = get_row(env.client, "P1")
    assert sum(row2["embedding"]) > 0.0


# ---------- pytest 入口（临时 lite 库） ----------


@pytest.fixture()
def lite_env(tmp_path):
    client = MilvusClient(str(tmp_path / "test_product_kb.db"))
    build_collection(client)
    env = build_env(client, {})
    yield env
    env.restore()  # 恢复被替换的 catalog_repo 函数，避免污染同进程后续测试
    client.close()


def test_upsert_same_id_no_duplicate(lite_env):
    case_upsert_same_id_no_duplicate(lite_env)


def test_bm25_refreshed_on_text_change(lite_env):
    case_bm25_refreshed_on_text_change(lite_env)


def test_scalar_filter_uses_new_value(lite_env):
    case_scalar_filter_uses_new_value(lite_env)


def test_delete_not_retrievable(lite_env):
    case_delete_not_retrievable(lite_env)


def test_degrade_then_recover(lite_env):
    case_degrade_then_recover(lite_env)
