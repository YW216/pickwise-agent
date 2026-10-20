"""第四步 HTTP 层验证（多租户隔离和fastapi接入.md §4.7）。

TestClient + 假 factory + 临时目录；不使用真实模型。覆盖：身份与 CSRF、
窗口归属隔离、chat/reset 生命周期、请求体校验、错误信封与状态码映射。
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.testing import create_test_app
from tests.test_session_manager import FakeAgent, FactoryRecorder

SECRET = "api-test-secret-key-0123456789abcdef"  # 40 字节


@pytest.fixture
def factory():
    return FactoryRecorder()


@pytest.fixture
def app(tmp_path, factory):
    return create_test_app(auth_secret_key=SECRET, session_dir=tmp_path,
                      factory=factory)


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def _init_user(client) -> dict:
    """完成匿名初始化，返回 {user_id, csrf_token}。"""
    resp = client.post("/api/auth/anonymous")
    assert resp.status_code == 200, resp.text
    return resp.json()["data"]


def _auth_headers(user) -> dict:
    return {"X-Pickwise-CSRF": user["csrf_token"]}


def _create_session(client, user) -> str:
    resp = client.post("/api/sessions", json={},
                       headers=_auth_headers(user))
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]["session_id"]


# ---------- 健康与匿名身份 ----------

def test_health_liveness(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_anonymous_issue_then_reuse_same_identity(client):
    first = _init_user(client)
    second = _init_user(client)
    assert first["user_id"] == second["user_id"]  # 复用身份，不签发新的
    assert first["csrf_token"] == second["csrf_token"]


def test_tampered_cookie_gets_401_and_clearing(client):
    client.cookies.set("pickwise_anon", "forged-token-value")
    resp = client.post("/api/auth/anonymous")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "AUTH_INVALID"
    assert "Max-Age=0" in resp.headers.get("set-cookie", "")  # 服务端清除失效 Cookie
    # 业务请求同样 401
    assert client.post("/api/sessions", json={}).status_code == 401


# ---------- 窗口创建、归属与隔离 ----------

def test_create_session_and_chat_roundtrip(client, factory):
    user = _init_user(client)
    session_id = _create_session(client, user)

    resp = client.post("/api/chat", json={"session_id": session_id,
                                          "message": "你好"},
                       headers=_auth_headers(user))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["session_id"] == session_id
    assert body["data"]["reply"] == "reply:你好"
    assert body["request_id"]

    resp = client.post("/api/chat", json={"session_id": session_id,
                                          "message": "再来"},
                       headers=_auth_headers(user))
    assert resp.json()["data"]["reply"] == "reply:再来"
    agent = factory.agents[f"{user['user_id']}:{session_id}"]
    assert len(agent.raw_messages) == 4  # 连续聊天恢复同一历史


def test_missing_csrf_is_403(client):
    user = _init_user(client)
    resp = client.post("/api/sessions", json={})  # 有 Cookie 无 CSRF 头
    assert resp.status_code == 403


def test_wrong_csrf_is_403(client):
    user = _init_user(client)
    resp = client.post("/api/sessions", json={},
                       headers={"X-Pickwise-CSRF": "wrong"})
    assert resp.status_code == 403


def test_another_users_window_is_404(client):
    user_a = _init_user(client)
    cookie_a = client.cookies.get("pickwise_anon")  # 保存 A 的凭证串
    session_a = _create_session(client, user_a)

    # B 是另一个匿名身份：清掉 Cookie 重新初始化
    client.cookies.clear()
    user_b = _init_user(client)
    assert user_b["user_id"] != user_a["user_id"]
    session_b = _create_session(client, user_b)

    # B 使用 A 的窗口编号：统一 404，不暴露存在性
    resp = client.post("/api/chat", json={"session_id": session_a,
                                          "message": "越权"},
                       headers=_auth_headers(user_b))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "SESSION_NOT_FOUND"

    # 反向同样隔离：恢复 A 的 Cookie，拿 B 的窗口
    client.cookies.set("pickwise_anon", cookie_a)
    resp = client.post("/api/chat", json={"session_id": session_b,
                                          "message": "反向"},
                       headers=_auth_headers(user_a))
    assert resp.status_code == 404


def test_unknown_window_uuid_is_404(client):
    user = _init_user(client)
    ghost = str(uuid.uuid4())
    resp = client.post("/api/chat", json={"session_id": ghost,
                                          "message": "hi"},
                       headers=_auth_headers(user))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "SESSION_NOT_FOUND"


# ---------- 请求体与参数校验 ----------

def test_extra_field_user_id_in_body_is_422(client):
    user = _init_user(client)
    resp = client.post("/api/sessions",
                       json={"user_id": "someone-else"},
                       headers=_auth_headers(user))
    assert resp.status_code == 422  # extra=forbid：正文不接受 user_id


def test_blank_message_is_422(client):
    user = _init_user(client)
    session_id = _create_session(client, user)
    resp = client.post("/api/chat", json={"session_id": session_id,
                                          "message": "   "},
                       headers=_auth_headers(user))
    assert resp.status_code == 422


def test_invalid_session_uuid_is_422(client):
    user = _init_user(client)
    resp = client.post("/api/chat", json={"session_id": "not-a-uuid",
                                          "message": "hi"},
                       headers=_auth_headers(user))
    assert resp.status_code == 422


def test_request_body_too_large_is_413(client):
    user = _init_user(client)
    big = "x" * (70 * 1024)
    resp = client.post("/api/chat", json={"session_id": str(uuid.uuid4()),
                                          "message": big},
                       headers=_auth_headers(user))
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "REQUEST_TOO_LARGE"


# ---------- 两个应用实例互不共享 ----------

def test_two_app_instances_do_not_share_managers(tmp_path, factory):
    app1 = create_test_app(auth_secret_key=SECRET, session_dir=tmp_path / "one")
    app2 = create_test_app(auth_secret_key=SECRET, session_dir=tmp_path / "two")
    with TestClient(app1) as c1, TestClient(app2) as c2:
        user1 = _init_user(c1)
        session1 = _create_session(c1, user1)
        user2 = _init_user(c2)
        # 同样的窗口编号在另一个 app 里不存在（各自目录 + 各自管理器）
        resp = c2.post("/api/chat", json={"session_id": session1,
                                          "message": "hi"},
                       headers=_auth_headers(user2))
        assert resp.status_code == 404
        assert app1.state.session_manager is not app2.state.session_manager


# ---------- 商品接口（依赖本地 PG：不可用则跳过） ----------

def test_product_detail_not_found_is_404(client):
    user = _init_user(client)
    resp = client.get("/api/products/NO-SUCH-ID",
                      headers=_auth_headers(user))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "PRODUCT_NOT_FOUND"


def test_favorite_requires_valid_session_ref(client):
    user = _init_user(client)
    headers = _auth_headers(user)
    session = client.post("/api/sessions", json={}, headers=headers)
    assert session.status_code == 201
    session_id = session.json()["data"]["session_id"]
    resp = client.post("/api/favorites/ref/BAD", headers=headers,
                       json={"session_id": session_id})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "PRODUCT_REF_NOT_FOUND"


def test_favorite_detail_uses_user_favorite_product_id(client, monkeypatch):
    user = _init_user(client)

    async def fake_lookup(database_url, user_id, product_id):
        assert user_id == user["user_id"]
        assert product_id == "LP-01"
        return product_id

    async def fake_detail(product_id, database_url):
        assert product_id == "LP-01"
        return {"success": True, "data": {"product": {
            "product_id": product_id,
            "name": "测试商品",
            "brand": "品牌",
            "category": "笔记本",
            "price": 100,
        }}}

    monkeypatch.setattr("app.api.routes.get_favorite_product_id", fake_lookup)
    monkeypatch.setattr("app.api.routes.get_detail", fake_detail)
    resp = client.get("/api/favorites/item/LP-01/detail",
                      headers=_auth_headers(user))

    assert resp.status_code == 200
    product = resp.json()["data"]["product"]
    assert product["name"] == "测试商品"
    assert "product_id" not in product


def test_favorites_response_contains_product_id_for_client_actions(client, monkeypatch):
    user = _init_user(client)

    async def fake_get_favorites(**kwargs):
        return {"success": True, "data": {
            "favorites": [{"product_id": "LP-01",
                           "name": "测试商品", "brand": "品牌", "category": "笔记本",
                           "price": 100, "summary": "测试"}],
            "total": 1,
        }}

    monkeypatch.setattr("app.api.routes.get_user_favorites", fake_get_favorites)
    resp = client.get("/api/favorites", headers=_auth_headers(user))
    assert resp.status_code == 200
    favorite = resp.json()["data"]["favorites"][0]
    assert favorite["name"] == "测试商品"
    assert favorite["product_id"] == "LP-01"


def test_remove_favorite_uses_user_scoped_product_id(client, monkeypatch):
    user = _init_user(client)
    calls = []

    async def fake_remove(database_url, user_id, product_id):
        calls.append((user_id, product_id))
        return True

    monkeypatch.setattr("app.api.routes.remove_favorite", fake_remove)
    resp = client.delete("/api/favorites/item/LP-01",
                         headers=_auth_headers(user))
    assert resp.status_code == 200
    assert resp.json()["data"]["favorited"] is False
    assert calls == [(user["user_id"], "LP-01")]


@pytest.mark.parametrize("action", ["detail", "remove"])
def test_favorite_actions_cannot_access_another_users_favorite(client, monkeypatch, action):
    owner = _init_user(client)
    favorites = {(owner["user_id"], "LP-01")}
    client.cookies.clear()
    other = _init_user(client)
    assert other["user_id"] != owner["user_id"]

    async def fake_lookup(database_url, user_id, product_id):
        return product_id if (user_id, product_id) in favorites else None

    async def fake_remove(database_url, user_id, product_id):
        key = (user_id, product_id)
        if key not in favorites:
            return False
        favorites.remove(key)
        return True

    async def unexpected_detail(*args, **kwargs):
        pytest.fail("Unowned favorite must not reach the product query")

    monkeypatch.setattr("app.api.routes.get_favorite_product_id", fake_lookup)
    monkeypatch.setattr("app.api.routes.remove_favorite", fake_remove)
    monkeypatch.setattr("app.api.routes.get_detail", unexpected_detail)
    if action == "detail":
        resp = client.get("/api/favorites/item/LP-01/detail")
    else:
        resp = client.delete("/api/favorites/item/LP-01", headers=_auth_headers(other))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "FAVORITE_NOT_FOUND"
    assert (owner["user_id"], "LP-01") in favorites
