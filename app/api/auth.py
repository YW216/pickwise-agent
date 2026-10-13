"""匿名身份凭证（多租户隔离和fastapi接入.md §4.2/§4.3）。

第一版不建注册登录表，用「服务端签名的匿名 Cookie」建立可信身份：
- 签名用 itsdangerous（成熟库，不手写加解密）；签名保证不可伪造，
  不做内容加密——payload 只放用户编号与 CSRF 随机值。
- 凭证内容：版本、用户 UUID、CSRF 随机值；签名带时间戳，统一有效期。
- 携带失效/被篡改 Cookie → 401 并由服务端用相同 Path 清除，不直接签发新身份。
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from uuid import uuid4

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

COOKIE_NAME = "pickwise_anon"
CSRF_HEADER = "X-Pickwise-CSRF"
_SALT = "pickwise.anon.v1"   # 换 salt = 全量凭证作废（版本化入口）
_PAYLOAD_VERSION = 1


class AuthError(Exception):
    """凭证缺失/非法/过期的统一异常（HTTP 层映射 401）。"""


@dataclass(frozen=True)
class AnonymousIdentity:
    """验签通过的可信身份：HTTP 层据此取得 user_id 与 CSRF 比对值。"""

    user_id: str
    csrf_token: str
    issued_at: float
    expires_at: float


class AnonymousAuth:
    """签发与验证匿名凭证（由应用持有；密钥来自配置，测试可注入）。"""

    def __init__(self, secret_key: str, max_age_seconds: float):
        if len(secret_key.encode()) < 32:
            raise RuntimeError("AUTH_SECRET_KEY 至少 32 字节高强度随机材料")
        self._max_age = float(max_age_seconds)
        self._serializer = URLSafeTimedSerializer(secret_key, salt=_SALT)

    @property
    def max_age_seconds(self) -> float:
        return self._max_age

    def issue(self) -> tuple[AnonymousIdentity, str]:
        """签发新身份，返回 (身份, 签名串)。"""
        now = time.time()
        payload = {"v": _PAYLOAD_VERSION,
                   "user_id": str(uuid4()),
                   "csrf": secrets.token_urlsafe(32)}
        signed = self._serializer.dumps(payload)
        identity = AnonymousIdentity(
            user_id=payload["user_id"], csrf_token=payload["csrf"],
            issued_at=now, expires_at=now + self._max_age,
        )
        return identity, signed

    def verify(self, signed: str) -> AnonymousIdentity:
        """验签并检查期限；返回可信身份，失败抛 AuthError。"""
        try:
            payload, issued_at = self._serializer.loads(
                signed, max_age=self._max_age, return_timestamp=True,
            )
        except SignatureExpired as exc:
            raise AuthError("匿名凭证已过期") from exc
        except BadSignature as exc:
            raise AuthError("匿名凭证非法") from exc
        if payload.get("v") != _PAYLOAD_VERSION:
            raise AuthError("匿名凭证版本不识别")
        user_id, csrf = payload.get("user_id"), payload.get("csrf")
        if not isinstance(user_id, str) or not isinstance(csrf, str):
            raise AuthError("匿名凭证内容不完整")
        return AnonymousIdentity(
            user_id=user_id, csrf_token=csrf,
            issued_at=issued_at.timestamp(), expires_at=issued_at.timestamp() + self._max_age,
        )

    def verify_csrf(self, identity: AnonymousIdentity, header_value: str | None,
                    origin: str | None, allowed_origins: list[str]) -> None:
        """写请求的 CSRF 与来源校验（§4.3）。

        - X-Pickwise-CSRF 必须与凭证内随机值安全比对一致；
        - Origin 头存在时必须在允许列表内（同源请求浏览器可不带 Origin；
          非浏览器客户端缺少 Origin 不构成豁免——CSRF 头仍然必查）。
        """
        if not header_value or not secrets.compare_digest(header_value, identity.csrf_token):
            raise AuthError("CSRF 校验失败")
        if origin and allowed_origins and origin not in allowed_origins:
            raise AuthError("来源不在允许列表内")
