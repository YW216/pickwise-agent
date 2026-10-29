"""Password hashing and server-side authentication service tests."""

import asyncio

import pytest

from app.api.auth import AuthError, AuthService, EmailAlreadyRegistered, hash_password, verify_password
from tests.in_memory_auth_store import InMemoryAuthStore


def test_password_hash_roundtrip_and_salt_uniqueness():
    first = hash_password("a-long-test-password")
    second = hash_password("a-long-test-password")
    assert first != second
    assert verify_password("a-long-test-password", first)
    assert not verify_password("wrong-password", first)


def test_password_policy():
    with pytest.raises(AuthError):
        hash_password("short")
    assert not verify_password("anything", "not-a-password-hash")


def test_register_login_authenticate_and_logout():
    async def run():
        service = AuthService(InMemoryAuthStore(), session_ttl_seconds=3600)
        user, token, csrf, expires = await service.register(
            "Student@Example.com", "a-long-test-password", "Student",
        )
        assert expires > 0
        current = await service.authenticate(token)
        assert current is not None
        assert current[0] == user
        assert current[1] == csrf

        logged_in = await service.login("student@example.com", "a-long-test-password")
        assert logged_in[0].user_id == user.user_id
        assert logged_in[1] != token

        await service.logout(token)
        assert await service.authenticate(token) is None

    asyncio.run(run())


def test_duplicate_email_and_invalid_credentials():
    async def run():
        service = AuthService(InMemoryAuthStore(), session_ttl_seconds=3600)
        await service.register("student@example.com", "a-long-test-password", "Student")
        with pytest.raises(EmailAlreadyRegistered):
            await service.register("STUDENT@example.com", "a-long-test-password", "Another")
        with pytest.raises(AuthError):
            await service.login("student@example.com", "incorrect-password")

    asyncio.run(run())


def test_session_token_is_opaque_and_only_its_hash_is_stored():
    async def run():
        store = InMemoryAuthStore()
        service = AuthService(store, session_ttl_seconds=3600)
        _, token, _, _ = await service.register(
            "opaque@example.com", "a-long-test-password", "Opaque",
        )
        assert token not in store._sessions
        assert len(store._sessions) == 1

    asyncio.run(run())
