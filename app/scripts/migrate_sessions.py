"""把旧 Web 本地 JSON 会话迁移到 PostgreSQL sessions.messages。

用法：
  python app/scripts/migrate_sessions.py --dry-run
  python app/scripts/migrate_sessions.py

迁移是幂等的：目标会话已经存在时跳过，不覆盖 PostgreSQL 中已有状态。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.storage import load_session  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.db.connection import close_all_pools  # noqa: E402
from app.db.session_repo import PostgresSessionStore  # noqa: E402
from app.db.session_schema import ensure_session_schema  # noqa: E402
from app.identity import SessionIdentity  # noqa: E402
from app.services.session_store import SessionMessage, SessionState  # noqa: E402


def _title(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "user" and str(message.get("content", "")).strip():
            return str(message["content"]).strip().replace("\n", " ")[:40]
    return "新会话"


def discover_sessions(root: Path) -> list[tuple[Path, SessionIdentity]]:
    """扫描 users/{user_id}/{session_id}.json，忽略 memory 目录。"""
    found: list[tuple[Path, SessionIdentity]] = []
    users = root / "users"
    if not users.exists():
        return found
    for path in sorted(users.glob("*/*.json")):
        try:
            identity = SessionIdentity(path.parent.name, path.stem)
        except ValueError:
            continue
        found.append((path, identity))
    return found


def _to_messages(identity: SessionIdentity, raw: list[dict]) -> tuple[SessionMessage, ...]:
    messages: list[SessionMessage] = []
    for item in raw:
        role = item.get("role")
        if role not in {"user", "assistant", "tool"}:
            continue
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"role", "content", "tool_call_id"}
        }
        content = item.get("content")
        messages.append(
            SessionMessage(
                role=role,
                content=content if isinstance(content, str) else "",
                tool_call_id=item.get("tool_call_id"),
                metadata=metadata,
            )
        )
    return tuple(messages)


def read_legacy(path: Path, identity: SessionIdentity) -> SessionState:
    loaded = load_session(str(path), identity=identity)
    if loaded is None:
        raise ValueError("会话文件不存在、格式过期或内容不合法")
    stat = path.stat()
    created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
    updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    messages = _to_messages(identity, loaded["messages"])
    return SessionState(
        user_id=identity.user_id,
        session_id=identity.session_id,
        title=_title(loaded["messages"]),
        summary=loaded.get("summary"),
        short_term_memory=loaded.get("short_term_memory") or {},
        products=loaded.get("products") or {},
        version=1,
        status="active",
        created_at=created_at,
        updated_at=updated_at,
        last_message_at=updated_at if messages else None,
        messages=messages,
    )


async def migrate(root: Path, dry_run: bool) -> dict[str, int]:
    candidates = discover_sessions(root)
    stats = {"found": len(candidates), "migrated": 0, "skipped": 0, "failed": 0}
    if dry_run:
        for path, identity in candidates:
            try:
                state = read_legacy(path, identity)
                print(f"[dry-run] {identity.user_id}/{identity.session_id}: {len(state.messages)} 条消息")
            except Exception as exc:
                stats["failed"] += 1
                print(f"[failed] {path}: {type(exc).__name__}: {exc}")
        return stats

    ensure_session_schema(settings.database_url)
    store = PostgresSessionStore(settings.database_url)
    for path, identity in candidates:
        try:
            state = read_legacy(path, identity)
            existing = await store.load_session(identity.user_id, identity.session_id)
            if existing is not None:
                stats["skipped"] += 1
                continue
            created = await store.create_session(
                identity.user_id, identity.session_id, title=state.title,
            )
            state = SessionState(
                **{**state.__dict__, "created_at": created.created_at,
                   "updated_at": created.updated_at, "version": created.version}
            )
            await store.save_session(state, expected_version=created.version)
            stats["migrated"] += 1
        except Exception as exc:
            stats["failed"] += 1
            print(f"[failed] {path}: {type(exc).__name__}: {exc}")
    await close_all_pools()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="迁移 Web 本地 JSON 会话到 PostgreSQL")
    parser.add_argument("--root", default=settings.web_session_dir, help="旧会话根目录")
    parser.add_argument("--dry-run", action="store_true", help="只扫描和校验，不写数据库")
    args = parser.parse_args()
    stats = asyncio.run(migrate(Path(args.root), args.dry_run))
    print(json.dumps(stats, ensure_ascii=False))
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
