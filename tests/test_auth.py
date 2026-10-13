"""匿名身份凭证单元测试（多租户隔离和fastapi接入.md §4.2/§4.3）。"""

import pytest
from itsdangerous import URLSafeTimedSerializer

from app.api.auth import AuthError, AnonymousAuth


SECRET = "unit-test-secret-key-0123456789abcdef"  # 40 字节，满足下限


def test_issue_and_verify_roundtrip():
    auth = AnonymousAuth(SECRET, max_age_seconds=86400)
    identity, signed = auth.issue()
    verified = auth.verify(signed)
    assert verified.user_id == identity.user_id
    assert verified.csrf_token == identity.csrf_token
    assert verified.expires_at > verified.issued_at


def test_tampered_cookie_is_rejected():
    auth = AnonymousAuth(SECRET, max_age_seconds=86400)
    _, signed = auth.issue()
    with pytest.raises(AuthError):
        auth.verify(signed[:-4] + "hacked")
    with pytest.raises(AuthError):
        auth.verify("totally-not-a-token")


def test_expired_cookie_is_rejected_without_sleep():
    # 负有效期：签发即过期（itsdangerous 按签名时间戳判期，无需真实等待）
    auth = AnonymousAuth(SECRET, max_age_seconds=-1)
    _, signed = auth.issue()
    with pytest.raises(AuthError):
        auth.verify(signed)


def test_unknown_payload_version_is_rejected():
    other = URLSafeTimedSerializer(SECRET, salt="pickwise.anon.v1")
    forged = other.dumps({"v": 99, "user_id": "u", "csrf": "c"})
    auth = AnonymousAuth(SECRET, max_age_seconds=86400)
    with pytest.raises(AuthError):
        auth.verify(forged)


def test_weak_secret_is_rejected():
    with pytest.raises(RuntimeError):
        AnonymousAuth("short", max_age_seconds=86400)


def test_csrf_check():
    auth = AnonymousAuth(SECRET, max_age_seconds=86400)
    identity, _ = auth.issue()
    # 正确
    auth.verify_csrf(identity, identity.csrf_token, None, [])
    # 缺失/错误 → AuthError
    with pytest.raises(AuthError):
        auth.verify_csrf(identity, None, None, [])
    with pytest.raises(AuthError):
        auth.verify_csrf(identity, "wrong-token", None, [])
    # 非允许 Origin → AuthError；允许列表为空时不做 Origin 检查（同源开发态）
    with pytest.raises(AuthError):
        auth.verify_csrf(identity, identity.csrf_token,
                         "https://evil.example", ["https://app.example"])
    auth.verify_csrf(identity, identity.csrf_token,
                     "https://app.example", ["https://app.example"])


def test_csrf_comparison_is_not_vulnerable_to_prefix_match():
    auth = AnonymousAuth(SECRET, max_age_seconds=86400)
    identity, _ = auth.issue()
    with pytest.raises(AuthError):
        auth.verify_csrf(identity, identity.csrf_token[:8], None, [])
