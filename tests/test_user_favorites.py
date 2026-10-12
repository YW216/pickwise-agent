"""真实 PostgreSQL 临时数据库验证收藏隔离和迁移，结束后删除测试库。"""

from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import pytest

from app.agent.tools.manager import ToolManager
from app.config.settings import settings
from app.db.catalog_repo import init_schema_and_seed
from app.db.user_repo import add_favorite, get_favorites, init_user_schema, remove_favorite


@pytest.fixture(scope="module")
def user_database():
    # 显式新建临时库；不会在开发数据库里建立试验表或修改真实收藏。
    admin_url = make_conninfo(settings.database_url, dbname="postgres", connect_timeout=3)
    try:
        admin = psycopg.connect(admin_url, autocommit=True)
    except psycopg.Error:
        pytest.skip("PostgreSQL 不可用，跳过临时数据库集成验证")
    name = f"pickwise_isolation_test_{uuid4().hex}"
    url = make_conninfo(settings.database_url, dbname=name, connect_timeout=3)
    try:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    except psycopg.errors.InsufficientPrivilege:
        admin.close()
        pytest.skip("测试连接没有创建临时数据库的权限")
    try:
        products = {
            pid: {"product_id": pid, "name": f"测试商品 {pid}", "brand": "测试品牌",
                  "category": category, "price": price, "specs": {}, "introduction": ["介绍"]}
            for pid, category, price in (("P1", "笔记本", 5000), ("P2", "手机", 3000))
        }
        init_schema_and_seed(url, {"products": products, "favorites": ["P1", "P2"]})
        assert init_user_schema(url) == 2
        yield url
    finally:
        # name 只能来自上面固定前缀和本次 UUID，从未使用用户输入作为删除目标。
        admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.close()


def test_migration_preserves_legacy_owner_and_never_resurrects_deleted_favorites(user_database):
    url = user_database
    with psycopg.connect(url) as conn:
        legacy = conn.execute("SELECT product_id, added_at FROM favorites ORDER BY product_id").fetchall()
        migrated = conn.execute(
            "SELECT product_id, added_at FROM user_favorites WHERE user_id = 'default' ORDER BY product_id"
        ).fetchall()
    assert migrated == legacy
    assert get_favorites(url, str(uuid4())) == []
    assert remove_favorite(url, "default", "P1")
    assert init_user_schema(url) == 0
    assert [row["product_id"] for row in get_favorites(url, "default")] == ["P2"]
    with psycopg.connect(url) as conn:
        assert conn.execute("SELECT product_id, added_at FROM favorites ORDER BY product_id").fetchall() == legacy


def test_users_can_favorite_same_product_and_delete_independently(user_database):
    url, a, b = user_database, str(uuid4()), str(uuid4())
    assert add_favorite(url, a, "P1")
    assert not add_favorite(url, a, "P1")
    assert add_favorite(url, b, "P1")
    assert remove_favorite(url, a, "P1")
    assert not remove_favorite(url, a, "P1")
    assert get_favorites(url, a) == []
    assert [item["product_id"] for item in get_favorites(url, b)] == ["P1"]


def test_bound_favorite_tool_ignores_model_supplied_user_id(user_database):
    url, a, b = user_database, str(uuid4()), str(uuid4())
    add_favorite(url, a, "P1")
    add_favorite(url, b, "P2")
    a_tools = ToolManager(allowed_tools={"get_user_favorites"}, user_id=a, database_url=url)
    b_tools = ToolManager(allowed_tools={"get_user_favorites"}, user_id=b, database_url=url)
    result = a_tools.execute_tool("get_user_favorites", {"user_id": b, "database_url": "other"})
    assert result["success"]
    assert [item["product_id"] for item in result["data"]["favorites"]] == ["P1"]
    assert "notice" in result
    assert [item["product_id"] for item in b_tools.execute_tool("get_user_favorites", {})["data"]["favorites"]] == ["P2"]
    assert not b_tools.execute_tool("get_user_favorites", {"category": "笔记本"})["success"]


def test_favorites_use_current_product_price(user_database):
    url, user_id = user_database, str(uuid4())
    add_favorite(url, user_id, "P1")
    with psycopg.connect(url) as conn:
        conn.execute("UPDATE products SET price = 4500 WHERE product_id = 'P1'")
    assert get_favorites(url, user_id)[0]["price"] == 4500


@pytest.mark.parametrize("user_id", ["", "../default", None])
def test_invalid_identity_is_rejected_before_database_access(user_id):
    for operation, args in (
        (get_favorites, ("unused", user_id)),
        (add_favorite, ("unused", user_id, "P1")),
        (remove_favorite, ("unused", user_id, "P1")),
    ):
        with pytest.raises(ValueError):
            operation(*args)


def test_missing_identity_fails_without_legacy_fallback():
    tools = ToolManager(allowed_tools={"get_user_favorites"}, database_url="unused")
    result = tools.execute_tool("get_user_favorites", {})
    assert not result["success"]
    assert "上下文" in result["error"]
