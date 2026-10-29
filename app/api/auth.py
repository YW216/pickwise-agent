"""Server-side password authentication and login sessions."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

COOKIE_NAME = "pickwise_session"
CSRF_HEADER = "X-Pickwise-CSRF"


class AuthError(Exception):
    """Authentication or credential validation failure."""


class EmailAlreadyRegistered(AuthError):
    pass


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    email: str
    display_name: str


@dataclass(frozen=True)
class LoginSession:
    user: AuthenticatedUser
    csrf_token: str
    expires_at: float


class AuthStore(ABC):
    @abstractmethod
    async def create_user(self, email: str, password_hash: str, display_name: str) -> AuthenticatedUser: ...
    @abstractmethod
    async def get_user_by_email(self, email: str) -> tuple[AuthenticatedUser, str] | None: ...
    @abstractmethod
    async def create_session(self, token_hash: str, user_id: str, csrf_token: str, expires_at: float) -> None: ...
    @abstractmethod
    async def get_session(self, token_hash: str) -> LoginSession | None: ...
    @abstractmethod
    async def revoke_session(self, token_hash: str) -> None: ...


def normalize_email(email: str) -> str:
    value = email.strip().lower()
    if len(value) > 254 or value.count("@") != 1:
        raise AuthError("邮箱格式不正确")
    local, domain = value.rsplit("@", 1)
    if not local or not domain or "." not in domain or any(ch.isspace() for ch in value):
        raise AuthError("邮箱格式不正确")
    return value


def hash_password(password: str) -> str:
    if not isinstance(password, str) or not 10 <= len(password) <= 128:
        raise AuthError("密码长度需为 10 到 128 个字符")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return "scrypt:16384:8:1:{}:{}".format(salt.hex(), digest.hex())


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_hex, digest_hex = encoded.split(":")
        if algorithm != "scrypt":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, MemoryError):
        return False


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthService:
    def __init__(self, store: AuthStore, session_ttl_seconds: int):
        self._store = store
        self._session_ttl_seconds = session_ttl_seconds

    async def register(self, email: str, password: str, display_name: str):
        normalized = normalize_email(email)
        name = display_name.strip()
        if not 1 <= len(name) <= 80:
            raise AuthError("昵称长度需为 1 到 80 个字符")
        encoded = await asyncio.to_thread(hash_password, password)
        user = await self._store.create_user(normalized, encoded, name)
        return await self._new_session(user)

    async def login(self, email: str, password: str):
        normalized = normalize_email(email)
        record = await self._store.get_user_by_email(normalized)
        if record is None:
            await asyncio.to_thread(hash_password, "invalid-login-padding")
            raise AuthError("邮箱或密码不正确")
        user, encoded = record
        if not await asyncio.to_thread(verify_password, password, encoded):
            raise AuthError("邮箱或密码不正确")
        return await self._new_session(user)

    async def authenticate(self, token: str | None):
        if not token:
            return None
        session = await self._store.get_session(hash_session_token(token))
        if session is None or session.expires_at <= time.time():
            return None
        return session.user, session.csrf_token, session.expires_at

    async def logout(self, token: str | None) -> None:
        if token:
            await self._store.revoke_session(hash_session_token(token))

    async def _new_session(self, user: AuthenticatedUser):
        token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        expires_at = time.time() + self._session_ttl_seconds
        await self._store.create_session(hash_session_token(token), user.user_id, csrf_token, expires_at)
        return user, token, csrf_token, expires_at
