"""内部用户、会话身份及存储路径；HTTP 层负责验证身份凭证。"""

from dataclasses import dataclass
from pathlib import Path
import re
from uuid import UUID


class SessionOwnershipError(ValueError):
    """文件身份与调用方不匹配，拒绝读取、覆盖或删除。"""


def validate_user_id(user_id: str) -> str:
    """兼容 CLI 的 default 等内部编号，禁止路径分隔符和空身份。"""
    if not isinstance(user_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", user_id):
        raise ValueError("用户编号不合法")
    return user_id


@dataclass(frozen=True)
class SessionIdentity:
    """Web 用户与聊天窗口使用独立 UUID，规范化后用于文件归属校验。"""

    user_id: str
    session_id: str

    def __post_init__(self):
        for name in ("user_id", "session_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} 必须是 UUID")
            try:
                normalized = str(UUID(value))
            except ValueError:
                raise ValueError(f"{name} 必须是 UUID") from None
            object.__setattr__(self, name, normalized)

    def session_path(self, root: str | Path) -> Path:
        return self._path(root, "users", self.user_id, f"{self.session_id}.json")

    def memory_dir(self, root: str | Path) -> Path:
        return self._path(root, "memory")

    @staticmethod
    def _path(root: str | Path, *parts: str) -> Path:
        base = Path(root).resolve()
        target = base.joinpath(*parts).resolve()
        if not target.is_relative_to(base):
            raise ValueError("用户数据路径超出存储目录")
        return target
