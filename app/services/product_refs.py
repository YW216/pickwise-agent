"""当前会话推荐商品的展示引用。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from secrets import token_urlsafe


@dataclass
class _RefRecord:
    user_id: str
    session_id: str
    product_id: str
    display_no: int
    last_seen: float
    name: str = ""


_refs: dict[str, _RefRecord] = {}
_numbers: dict[tuple[str, str, str], int] = {}
_next_numbers: dict[tuple[str, str], int] = {}
_TTL_SECONDS = 3600
_MAX_REFS = 2000


def _cleanup() -> None:
    cutoff = time.time() - _TTL_SECONDS
    for ref, record in list(_refs.items()):
        if record.last_seen < cutoff:
            _refs.pop(ref, None)
            _numbers.pop((record.user_id, record.session_id, record.product_id), None)


def remember_product(
    user_id: str, session_id: str, product_id: str, name: str = "",
    display_no: int | None = None,
) -> tuple[str, int]:
    """登记并返回该商品的展示引用；同一商品在同一会话内复用编号。

    ``display_no`` 只在从落盘快照恢复时传入：编号必须与历史一致，
    同时把会话的计数器推到该编号之后，避免新商品撞号。
    """
    _cleanup()
    key = (user_id, session_id, product_id)
    existing_no = _numbers.get(key)
    if existing_no is not None:
        display_no = existing_no
    elif display_no is None:
        display_no = _next_numbers.get((user_id, session_id), 0) + 1
    if display_no <= 0:
        display_no = _next_numbers.get((user_id, session_id), 0) + 1
    _numbers[key] = display_no
    session_key = (user_id, session_id)
    _next_numbers[session_key] = max(_next_numbers.get(session_key, 0), display_no)
    ref = next((ref for ref, item in _refs.items()
                if (item.user_id, item.session_id, item.product_id) == key), None)
    existing = _refs.get(ref) if ref is not None else None
    if ref is None:
        ref = token_urlsafe(18)
    _refs[ref] = _RefRecord(
        user_id, session_id, product_id, display_no, time.time(),
        name=name or (existing.name if existing is not None else ""),
    )
    if len(_refs) > _MAX_REFS:
        oldest = min(_refs, key=lambda item: _refs[item].last_seen)
        _refs.pop(oldest, None)
    return ref, display_no


def _locate(user_id: str, session_id: str, ref: str) -> _RefRecord | None:
    """按 (用户, 会话) 作用域取记录并刷新活跃时间；跨会话一律视为不存在。"""
    _cleanup()
    record = _refs.get(ref)
    if record is None or (record.user_id, record.session_id) != (user_id, session_id):
        return None
    record.last_seen = time.time()
    return record


def resolve_product_ref(user_id: str, session_id: str, ref: str) -> str | None:
    record = _locate(user_id, session_id, ref)
    return record.product_id if record is not None else None


def resolve_product_context(user_id: str, session_id: str, ref: str) -> dict | None:
    """聊天上下文需要的商品标识与名称；名称缺失时由调用方降级显示。"""
    record = _locate(user_id, session_id, ref)
    if record is None:
        return None
    return {
        "ref": ref,
        "product_id": record.product_id,
        "name": record.name,
        "display_no": record.display_no,
    }
