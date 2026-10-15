"""P2-1 商品读取路径验证：详情/对比直查 PG，改价即时生效。

行为锚点（develop_docs/多租户隔离和fastapi接入.md §2.2）：
- 改价后同一 ToolManager 查详情/对比取得新价——读当前值，不读启动快照
- compare 去重保序；缺商品明确返回缺失 ID，不静默少比一款
- 详情/对比模块零快照依赖（静态保证：源码不引用 app.db.snapshot）
"""

from pathlib import Path
from uuid import uuid4

import asyncio

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import pytest

from app.agent.tools import detail
from app.agent.tools.detail import compare_products, get_detail
from app.agent.tools.manager import ToolManager
from app.config.settings import settings
from app.db.catalog_repo import init_schema_and_seed


@pytest.fixture(scope="module")
def product_database():
    """临时 PG 库：三款商品（P2/P1 同品类带重叠参数、P3 无参数），结束后强删。"""
    admin_url = make_conninfo(settings.database_url, dbname="postgres", connect_timeout=3)
    try:
        admin = psycopg.connect(admin_url, autocommit=True)
    except psycopg.Error:
        pytest.skip("PostgreSQL 不可用，跳过临时数据库集成验证")
    name = f"pickwise_product_reads_test_{uuid4().hex}"
    url = make_conninfo(settings.database_url, dbname=name, connect_timeout=3)
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    except psycopg.errors.InsufficientPrivilege:
        admin.close()
        pytest.skip("测试连接没有创建临时数据库的权限")
    try:
        products = {
            pid: {"product_id": pid, "name": f"测试商品 {pid}", "brand": "测试品牌",
                  "category": category, "price": price, "specs": specs,
                  "introduction": ["介绍"]}
            for pid, category, price, specs in (
                ("P1", "笔记本", 5000, {"内存": "16GB"}),
                ("P2", "笔记本", 7000, {"内存": "32GB", "显卡": "独显"}),
                ("P3", "手机", 3000, {}),
            )
        }
        init_schema_and_seed(url, {"products": products, "favorites": []})
        yield url
    finally:
        # name 只能来自上面固定前缀和本次 UUID，从未使用用户输入作为删除目标。
        admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()


def set_price(url: str, pid: str, price: int) -> None:
    """直接改 PG 价格（模拟业务侧数据变更，工具层应即时可见）。"""
    with psycopg.connect(url) as conn:
        conn.execute("UPDATE products SET price = %s WHERE product_id = %s", (price, pid))


def test_get_detail_returns_current_price(product_database):
    url = product_database
    assert asyncio.run(get_detail("P1", database_url=url))["data"]["product"]["price"] == 5000
    set_price(url, "P1", 4500)  # 业务改价后，同一函数再次查询必须拿到新价
    assert asyncio.run(get_detail("P1", database_url=url))["data"]["product"]["price"] == 4500


def test_get_detail_missing_product_fails(product_database):
    result = asyncio.run(get_detail("GHOST", database_url=product_database))
    assert not result["success"]
    assert result["error"] == "商品不存在"


def test_get_detail_unreachable_database_fails_explicitly():
    # 127.0.0.1:1 本机内核立即拒绝，不依赖网络环境
    result = asyncio.run(get_detail("P1", database_url="postgresql://x:x@127.0.0.1:1/none"))
    assert not result["success"]
    assert "暂时不可用" in result["error"]


def test_compare_dedupes_and_keeps_input_order(product_database):
    url = product_database
    set_price(url, "P1", 5000)  # 自设确定状态：前面用例可能改过价（同库共用）
    result = asyncio.run(compare_products(["P2", "P1", "P2"], database_url=url))
    assert result["success"]
    data = result["data"]
    # 重复 ID 只比一次，且顺序跟输入走（批量返回的 dict 不决定展示顺序）
    assert [p["product_id"] for p in data["products"]] == ["P2", "P1"]
    price_field = data["fields"][0]
    assert price_field["field"] == "价格"
    assert price_field["values"] == {"P2": 7000, "P1": 5000}
    by_field = {f["field"]: f["values"] for f in data["fields"]}
    assert by_field["内存"] == {"P2": "32GB", "P1": "16GB"}
    assert by_field["显卡"] == {"P2": "独显", "P1": "-"}  # 单边参数补 "-"


def test_compare_missing_ids_fail_explicitly(product_database):
    result = asyncio.run(compare_products(["P1", "GHOST", "P3"], database_url=product_database))
    assert not result["success"]
    assert result["data"]["missing_ids"] == ["GHOST"]  # 不静默少比一款


def test_compare_all_dedup_below_two_fails(product_database):
    result = asyncio.run(compare_products(["P1", "P1"], database_url=product_database))
    assert not result["success"]


def test_tool_manager_reads_current_price(product_database):
    """完整执行链路：ToolManager（bind 焊死实例地址）同样读到 PG 当前价。"""
    url = product_database
    tools = ToolManager(
        allowed_tools={"get_detail", "compare_products"}, database_url=url,
    )
    set_price(url, "P1", 4500)
    detail_result = tools.execute_tool("get_detail", {"product_id": "P1"})
    assert detail_result["success"]
    assert detail_result["data"]["product"]["price"] == 4500

    cmp_result = tools.execute_tool("compare_products", {"product_ids": ["P1", "P3"]})
    assert cmp_result["success"]
    assert cmp_result["data"]["fields"][0]["values"] == {"P1": 4500, "P3": 3000}


def test_tool_manager_binds_instance_database_for_search(product_database):
    """P2-2：bind 层焊死实例地址——搜索只看实例库，与全局配置指向的库隔离。"""
    tools = ToolManager(allowed_tools={"search_catalog"}, database_url=product_database)
    result = tools.execute_tool("search_catalog", {"category": "笔记本"})
    assert result["success"]
    ids = [c["product_id"] for c in result["data"]["candidates"]]
    assert ids == ["P1", "P2"]  # 只来自临时库；若走了全局配置会混入开发库商品


# ---------- P2-3：硬条件按 PG 当前真值下推 ID 集合（假检索器验证编排） ----------

class _FakeClient:
    """最小 Milvus 客户端替身：只提供编排路径用到的两个方法。"""

    def has_collection(self, name):
        return True

    def load_collection(self, name):
        return None


class _FakeEmbedder:
    def encode_one(self, text):
        return [0.0]


def _patch_milvus(monkeypatch, fake_hybrid):
    """把 search_products 的检索面整体替换为假实现（不依赖真 Milvus）。"""
    import app.agent.tools.search_products as sp
    monkeypatch.setattr(sp, "_get_client", lambda: _FakeClient())
    monkeypatch.setattr(sp, "_get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(sp, "_hybrid_product_ids", fake_hybrid)


def test_search_products_filters_by_pg_current_ids(product_database, monkeypatch):
    """改价后按 PG 当前真值过滤：旧索引标量（若存在）不再漏召回/误杀。"""
    import app.agent.tools.search_products as sp

    url = product_database
    set_price(url, "P2", 3000)  # P2 笔记本降到 3000——真值进入预算，旧索引未必同步
    captured = {}

    def fake_hybrid(query, vec, top_k=10, recall_k=None, expr=None):
        captured["expr"] = expr
        return ["P2"]  # 与断言中的 expr 对应的窗口命中（一致性由断言保证）

    _patch_milvus(monkeypatch, fake_hybrid)
    result = sp.search_products(
        "轻薄笔记本", category="笔记本", max_price=3500, limit=5, database_url=url,
    )
    assert result["success"]
    # eligible 只含「笔记本 且 当前价 ≤3500」：P1(5000) 出局、P2(3000) 入围
    assert captured["expr"] == 'product_id in ["P2"]'
    assert [r["product_id"] for r in result["data"]["results"]] == ["P2"]


def test_search_products_empty_eligible_short_circuits(product_database, monkeypatch):
    """PG 硬条件无匹配 → 直接短路无匹配，不发起检索（空集不误当取消过滤）。"""
    import app.agent.tools.search_products as sp

    called = {"hybrid": False}

    def fake_hybrid(*args, **kwargs):
        called["hybrid"] = True
        return []

    _patch_milvus(monkeypatch, fake_hybrid)
    result = sp.search_products("任意", max_price=1, database_url=product_database)
    assert not result["success"]
    assert not called["hybrid"]


def test_search_products_truncates_to_limit(product_database, monkeypatch):
    """窗口召回（recall > limit）+ 按序截断：输出不超过 limit 且保持排序。"""
    import app.agent.tools.search_products as sp

    def fake_hybrid(query, vec, top_k=10, recall_k=None, expr=None):
        assert top_k == recall_k == 12  # limit=2 → recall=max(8,12)=12
        return ["P2", "P1", "P3"]  # 模拟相似度排序的窗口命中

    _patch_milvus(monkeypatch, fake_hybrid)
    result = sp.search_products("商品", limit=2, database_url=product_database)
    assert result["success"]
    assert [r["product_id"] for r in result["data"]["results"]] == ["P2", "P1"]


def test_search_products_skips_ids_missing_in_pg(product_database, monkeypatch):
    """索引残留、PG 已删的孤儿 ID：复核跳过不露出，槽位由后续命中补足。"""
    import app.agent.tools.search_products as sp

    def fake_hybrid(query, vec, top_k=10, recall_k=None, expr=None):
        return ["GHOST", "P1", "P3"]  # GHOST 只在索引里，PG 无此行

    _patch_milvus(monkeypatch, fake_hybrid)
    result = sp.search_products("商品", limit=2, database_url=product_database)
    assert result["success"]
    assert [r["product_id"] for r in result["data"]["results"]] == ["P1", "P3"]


def test_online_tool_modules_have_no_snapshot_import():
    """在线路径零快照依赖的静态保证：三个商品读取工具源码不得引用 snapshot。"""
    from app.agent.tools import catalog, search_products
    for module in (detail, catalog, search_products):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "snapshot" not in source, f"{module.__name__} 引用了快照"
