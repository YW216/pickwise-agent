"""进程内会话管理。

核心模型只有三层：

* ``(user_id, session_id)`` 是会话的唯一键；
* ``_entries`` 缓存当前进程正在使用的编排器；
* 每个窗口一把 ``asyncio.Lock``，保证同一窗口的 chat/reset 不同时修改
  同一个 ``MultiAgentOrchestrator``。

文件归属校验、TTL 回收和优雅关闭仍然保留；显式 FIFO 消费者、复杂状态机
和 SessionManager 专属线程池不再需要。同步文件操作和同步 factory 统一走
``run_in_executor``，不会阻塞 FastAPI 事件循环。
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from app.agent.storage import load_session, save_session
from app.identity import SessionIdentity, SessionOwnershipError, validate_user_id


class SessionManagerError(Exception):
    """SessionManager 对外失败的基类。"""

    code = "SESSION_ERROR"


class SessionManagerClosed(SessionManagerError):
    code = "SESSION_MANAGER_CLOSED"


class SessionNotFound(SessionManagerError):
    code = "SESSION_NOT_FOUND"


class SessionCreateBusy(SessionManagerError):
    code = "SESSION_CREATE_BUSY"


class SessionQueueFull(SessionManagerError):
    code = "SESSION_QUEUE_FULL"


class SessionCapacityFull(SessionManagerError):
    code = "SESSION_CAPACITY_FULL"


class SessionClosing(SessionManagerError):
    code = "SESSION_CLOSING"


@dataclass(frozen=True)
class ChatOutcome:
    reply: str
    failed: bool
    persisted: bool
    warnings: tuple[str, ...] = ()


@dataclass
class CleanupReport:
    reclaimed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


@dataclass
class ShutdownReport:
    closed: list[str] = field(default_factory=list)
    unsaved: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    timed_out: list[str] = field(default_factory=list)


@dataclass
class _Entry:
    """一个窗口在当前进程里的运行记录。"""

    identity: SessionIdentity
    last_completed_at: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    idle: asyncio.Event = field(default_factory=asyncio.Event)
    agent: Any = None
    pending: int = 0
    running: bool = False
    state: str = "active"  # active | closing | closed
    init_error: BaseException | None = None

    def __post_init__(self) -> None:
        self.idle.set()


class SessionManager:
    """管理 ``(user_id, session_id) -> MultiAgentOrchestrator``。

    ``max_pending_per_session`` 只限制等待同一窗口锁的请求数。
    ``max_parallel_jobs`` 限制同步文件/factory 操作，不限制异步模型调用；
    模型并发由应用级 LLM semaphore 管理。
    """

    def __init__(
        self,
        *,
        session_dir: str | Path,
        factory: Callable[[SessionIdentity], Any],
        clock: Callable[[], float] | None = None,
        ttl_seconds: float = 1800,
        max_sessions: int = 100,
        max_parallel_jobs: int = 4,
        max_pending_per_session: int = 8,
        max_pending_creates: int = 16,
        executor: Executor | None = None,
    ):
        self._session_dir = Path(session_dir)
        self._factory = factory
        self._clock = clock or time.monotonic
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._max_pending = max_pending_per_session
        self._executor = executor
        self._blocking_slots = asyncio.Semaphore(max_parallel_jobs)
        self._max_pending_creates = max_pending_creates

        self._entries: dict[tuple[str, str], _Entry] = {}
        self._registry_lock = asyncio.Lock()
        self._create_pending = 0
        self._closed = False
        self._cleanup_task: asyncio.Task | None = None

    # ---- 基础工具 ----

    def _check_open(self) -> None:
        if self._closed:
            raise SessionManagerClosed("SessionManager 已关闭")

    @staticmethod
    def _check_admittable(entry: _Entry) -> None:
        if entry.state != "active":
            raise SessionClosing("窗口正在关闭，请稍后重试")

    async def _run_blocking(self, fn: Callable[..., Any], *args: Any) -> Any:
        """把同步文件/factory 放到线程池，避免卡住事件循环。"""

        loop = asyncio.get_running_loop()
        async with self._blocking_slots:
            return await loop.run_in_executor(
                self._executor, lambda: fn(*args),
            )

    # ---- 创建窗口 ----

    async def create_session(self, user_id: str) -> SessionIdentity:
        self._check_open()
        validate_user_id(user_id)
        async with self._registry_lock:
            if self._create_pending >= self._max_pending_creates:
                raise SessionCreateBusy("窗口创建请求过多，请稍后重试")
            self._create_pending += 1
        try:
            return await self._run_blocking(self._create_session_file, user_id)
        finally:
            async with self._registry_lock:
                self._create_pending -= 1

    def _create_session_file(self, user_id: str) -> SessionIdentity:
        for _ in range(5):
            identity = SessionIdentity(user_id=user_id, session_id=str(uuid4()))
            path = identity.session_path(self._session_dir)
            if path.exists():
                continue
            try:
                save_session(str(path), [], None, identity=identity)
            except SessionOwnershipError:
                continue
            return identity
        raise SessionManagerError("无法分配新的窗口编号（连续冲突）")

    # ---- 定位窗口 ----

    async def _ensure_entry(self, identity: SessionIdentity) -> _Entry:
        self._check_open()
        key = (identity.user_id, identity.session_id)

        async with self._registry_lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._check_admittable(entry)
                return entry
            if len(self._entries) >= self._max_sessions:
                victim = self._oldest_idle_locked()
                if victim is None:
                    raise SessionCapacityFull(
                        f"运行窗口数已达上限（{self._max_sessions}）且无空闲可回收"
                    )
                victim.state = "closing"
            else:
                victim = None

        if victim is not None:
            error = await self._reclaim_one(victim)
            if error is not None:
                raise SessionCapacityFull(
                    f"运行窗口数已达上限，空闲窗口回收失败：{error}"
                )
            return await self._ensure_entry(identity)

        found = await self._run_blocking(self._verify_owned_file, identity)
        if not found:
            raise SessionNotFound("窗口不存在，请先创建")

        async with self._registry_lock:
            # 文件检查期间可能已有另一个请求完成创建。
            entry = self._entries.get(key)
            if entry is not None:
                self._check_admittable(entry)
                return entry
            if len(self._entries) >= self._max_sessions:
                raise SessionCapacityFull(
                    f"运行窗口数已达上限（{self._max_sessions}）且无空闲可回收"
                )
            entry = _Entry(identity=identity, last_completed_at=self._clock())
            self._entries[key] = entry
            return entry

    def _verify_owned_file(self, identity: SessionIdentity) -> bool:
        loaded = load_session(
            str(identity.session_path(self._session_dir)), identity=identity,
        )
        return loaded is not None

    def _oldest_idle_locked(self) -> _Entry | None:
        idle = [
            entry for entry in self._entries.values()
            if entry.state == "active"
            and not entry.running
            and entry.pending == 0
            and entry.agent is not None
        ]
        return min(idle, key=lambda item: item.last_completed_at) if idle else None

    # ---- chat / reset ----

    async def chat(self, identity: SessionIdentity, message: str) -> ChatOutcome:
        entry = await self._ensure_entry(identity)
        return await self._run_serialized(
            entry, lambda agent: self._chat_agent(agent, message),
        )

    async def reset(self, identity: SessionIdentity) -> None:
        entry = await self._ensure_entry(identity)

        async def operation(agent: Any) -> None:
            await self._run_blocking(agent.reset)

        await self._run_serialized(entry, operation)

    async def _run_serialized(
        self, entry: _Entry, operation: Callable[[Any], Any],
    ) -> Any:
        """用一把窗口锁串行执行；不同窗口不共享这把锁，因此可并行。"""

        queued = False
        acquired = False
        async with self._registry_lock:
            self._check_admittable(entry)
            if entry.running:
                if entry.pending >= self._max_pending:
                    raise SessionQueueFull(
                        f"窗口排队已达上限（{self._max_pending}）"
                    )
                entry.pending += 1
                queued = True
            entry.running = True
            entry.idle.clear()

        try:
            await entry.lock.acquire()
            acquired = True
            if queued:
                async with self._registry_lock:
                    entry.pending -= 1

            if entry.state != "active":
                raise entry.init_error or SessionClosing("窗口正在关闭，请稍后重试")

            if entry.agent is None:
                try:
                    entry.agent = await self._run_blocking(
                        self._factory, entry.identity,
                    )
                except Exception as exc:
                    entry.state = "closed"
                    entry.init_error = exc
                    async with self._registry_lock:
                        self._entries.pop(
                            (entry.identity.user_id, entry.identity.session_id),
                            None,
                        )
                    raise

            return await operation(entry.agent)
        finally:
            if not acquired:
                async with self._registry_lock:
                    if queued:
                        entry.pending -= 1
                        # 当前执行还没结束时，窗口仍然是忙的；只有队列已经
                        # 清空且锁也已释放，才可以让 cleanup/close 处理它。
                        if entry.pending == 0 and not entry.lock.locked():
                            entry.running = False
                            entry.idle.set()
                    else:
                        entry.running = False
                        entry.idle.set()
            else:
                async with self._registry_lock:
                    entry.last_completed_at = self._clock()
                    # 在注册表锁内释放窗口锁，避免下一个等待者先拿到窗口锁，
                    # 但后一个状态更新还没完成。
                    entry.lock.release()
                    if entry.pending == 0:
                        entry.running = False
                        entry.idle.set()
                    else:
                        # 还有等待者：保持 busy，避免 cleanup/close 抢先回收。
                        entry.running = True

    async def _chat_agent(self, agent: Any, message: str) -> ChatOutcome:
        reply = await agent.chat(message)
        failures = list(getattr(agent, "last_failures", []))
        failed = any(item.get("affects_answer") for item in failures)
        persisted = not any(item.get("stage") == "persistence" for item in failures)
        warnings = tuple(
            f"{item.get('stage', 'unknown')}_failed".upper()
            for item in failures if not item.get("affects_answer")
        )
        return ChatOutcome(
            reply=reply,
            failed=failed,
            persisted=persisted,
            warnings=warnings,
        )

    # ---- TTL 回收 ----

    async def cleanup(self) -> CleanupReport:
        report = CleanupReport()
        now = self._clock()
        async with self._registry_lock:
            expired = [
                entry for entry in self._entries.values()
                if entry.state == "active"
                and not entry.running
                and entry.pending == 0
                and entry.agent is not None
                and now - entry.last_completed_at >= self._ttl_seconds
            ]
            for entry in expired:
                entry.state = "closing"

        for entry in expired:
            error = await self._reclaim_one(entry)
            if error is None:
                report.reclaimed.append(entry.identity.session_id)
            else:
                report.failed[entry.identity.session_id] = error
        return report

    async def start_cleanup_loop(self, interval_seconds: float = 60.0) -> asyncio.Task:
        self._check_open()

        async def loop_body() -> None:
            while True:
                await asyncio.sleep(interval_seconds)
                try:
                    await self.cleanup()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # 周期清理失败不应终止 HTTP 服务。
                    pass

        self._cleanup_task = asyncio.create_task(loop_body())
        return self._cleanup_task

    async def _reclaim_one(self, entry: _Entry) -> str | None:
        try:
            await entry.idle.wait()
            if entry.agent is not None:
                await self._run_blocking(entry.agent.save)
                await entry.agent.close()
        except Exception as exc:
            entry.state = "active"
            return f"{type(exc).__name__}: {exc}"

        entry.state = "closed"
        async with self._registry_lock:
            self._entries.pop((entry.identity.user_id, entry.identity.session_id), None)
        return None

    # ---- 服务关闭 ----

    async def close(self, grace_seconds: float = 30.0) -> ShutdownReport:
        if self._closed:
            raise SessionManagerClosed("SessionManager 已关闭")
        self._closed = True
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None

        report = ShutdownReport()
        async with self._registry_lock:
            entries = list(self._entries.values())
            for entry in entries:
                entry.state = "closing"

        for entry in entries:
            try:
                await asyncio.wait_for(entry.idle.wait(), timeout=grace_seconds)
            except asyncio.TimeoutError:
                report.timed_out.append(entry.identity.session_id)
                report.unsaved.append(entry.identity.session_id)
                continue
            try:
                if entry.agent is not None:
                    await self._run_blocking(entry.agent.save)
                    await entry.agent.close()
            except Exception as exc:
                report.failed[entry.identity.session_id] = (
                    f"{type(exc).__name__}: {exc}"
                )
                report.unsaved.append(entry.identity.session_id)
                continue
            entry.state = "closed"
            async with self._registry_lock:
                self._entries.pop(
                    (entry.identity.user_id, entry.identity.session_id), None,
                )
            report.closed.append(entry.identity.session_id)
        return report

    def session_count(self) -> int:
        return len(self._entries)
