"""进程内会话管理。

核心模型只有三层：

* ``(user_id, session_id)`` 是会话的唯一键；
* ``_entries`` 缓存当前进程正在使用的编排器；
* 每个窗口一把 ``asyncio.Lock``，保证同一窗口的多个 chat 不同时修改
  同一个 ``MultiAgentOrchestrator``。

文件归属校验、TTL 回收和优雅关闭仍然保留；显式 FIFO 消费者、管理锁和
独立忙闲事件不再需要。窗口忙闲只看 ``entry.lock``；``closing`` 只用于
提示清理中的窗口。同步文件操作和同步 factory 统一走 ``asyncio.to_thread``，
不会阻塞 FastAPI 事件循环。管理方法只在同一个事件循环中调用；不含 await
的注册表和状态更新无需加锁。
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from app.agent.storage import load_session, save_session
from app.identity import SessionIdentity, SessionOwnershipError, validate_user_id
from app.services.session_service import SessionService, SessionServiceNotFound
from app.services.session_store import SessionMessage, SessionState
from app.services.chat_events import emit


class SessionManagerError(Exception):
    """SessionManager 对外失败的基类。"""

    code = "SESSION_ERROR"


class SessionManagerClosed(SessionManagerError):
    code = "SESSION_MANAGER_CLOSED"


class SessionNotFound(SessionManagerError):
    code = "SESSION_NOT_FOUND"


class SessionCapacityFull(SessionManagerError):
    code = "SESSION_CAPACITY_FULL"


class SessionBusy(SessionManagerError):
    code = "SESSION_BUSY"


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
    """一个窗口在当前进程里的运行记录。

    ``lock`` 就是"这个窗口忙不忙"的唯一信号：谁拿到锁，谁在用窗口。
    ``closing`` 只是提示新请求不要再进入；真正等待仍靠窗口锁。
    """

    identity: SessionIdentity
    last_completed_at: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    agent: Any = None
    busy: bool = False        # 本轮 chat 正在执行（同步检查并占用，忙时拒绝）
    save_failed: bool = False # 本轮持久化失败：TTL 暂不回收，保留内存中的最新数据
    closing: bool = False
    state: SessionState | None = None


class SessionManager:
    """管理 ``(user_id, session_id) -> MultiAgentOrchestrator``。

    同一窗口的请求直接等待窗口锁，不单独计数或限制等待请求数。
    同步文件/factory 走默认线程池，不限制异步模型调用；
    模型并发由应用级 LLM semaphore 管理。
    """

    def __init__(
        self,
        *,
        session_dir: str | Path,
        factory: Callable[[SessionIdentity], Any],
        session_service: SessionService | None = None,
        clock: Callable[[], float] | None = None,
        ttl_seconds: float = 1800,
        max_sessions: int = 100,
    ):
        self._session_dir = Path(session_dir)
        self._factory = factory
        self._session_service = session_service
        self._clock = clock or time.monotonic
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions

        self._entries: dict[tuple[str, str], _Entry] = {}
        self._closed = False
        self._cleanup_task: asyncio.Task | None = None

    # ---- 基础工具 ----

    def _check_open(self) -> None:
        if self._closed:
            raise SessionManagerClosed("SessionManager 已关闭")

    @staticmethod
    def _check_admittable(entry: _Entry) -> None:
        if entry.closing:
            raise SessionClosing("窗口正在关闭，请稍后重试")

    # ---- 创建窗口 ----

    async def create_session(self, user_id: str) -> SessionIdentity:
        self._check_open()
        validate_user_id(user_id)
        if self._session_service is not None:
            state = await self._session_service.create_session(user_id)
            return SessionIdentity(user_id=state.user_id, session_id=state.session_id)
        return await asyncio.to_thread(self._create_session_file, user_id)

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

    # ---- 定位运行窗口 ----
    # 首次聊天才创建 Entry；再次聊天复用同一个 Entry。
    async def _ensure_entry(self, identity: SessionIdentity) -> _Entry:
        self._check_open()
        key = (identity.user_id, identity.session_id)

        entry = self._entries.get(key)
        if entry is not None:
            self._check_admittable(entry)
            return entry
        # 满了直接拒绝，不在这里腾位：回收职责完全归 cleanup 循环
        # （60 秒周期自动回收超时窗口），避免 ensure 承担连锁回收。
        if len(self._entries) >= self._max_sessions:
            raise SessionCapacityFull(
                f"运行窗口数已达上限（{self._max_sessions}），请稍后重试"
            )

        # Entry 可以被回收。生产 Web 路径从 PostgreSQL 恢复快照；
        # CLI/测试路径继续校验本地 JSON 文件。
        state = None
        if self._session_service is not None:
            try:
                state = await self._session_service.get_session(
                    identity.user_id, identity.session_id,
                )
            except SessionServiceNotFound as exc:
                raise SessionNotFound("窗口不存在，请先创建") from exc
        else:
            found = await asyncio.to_thread(self._verify_owned_file, identity)
            if not found:
                raise SessionNotFound("窗口不存在，请先创建")

        # await 期间服务可能关闭、其他请求可能创建 Entry 或占满容量。
        self._check_open()
        entry = self._entries.get(key)
        if entry is not None:
            self._check_admittable(entry)
            return entry
        if len(self._entries) >= self._max_sessions:
            raise SessionCapacityFull(
                f"运行窗口数已达上限（{self._max_sessions}）且无空闲可回收"
            )
        entry = _Entry(
            identity=identity,
            last_completed_at=self._clock(),
            state=state,
        )
        self._entries[key] = entry
        return entry

    def _verify_owned_file(self, identity: SessionIdentity) -> bool:
        loaded = load_session(
            str(identity.session_path(self._session_dir)), identity=identity,
        )
        return loaded is not None

    # ---- chat ----

    async def chat(
        self, identity: SessionIdentity, message: str,
        product_context: list[dict] | None = None,
    ) -> ChatOutcome:
        entry = await self._ensure_entry(identity)
        outcome = await self._run_serialized(
            entry, lambda _agent: self._chat_entry(entry, message, product_context=product_context),
        )
        # 持久化失败标记：TTL 暂不回收该窗口，避免丢失内存中的最新数据
        entry.save_failed = not outcome.persisted
        return outcome

    async def chat_stream(
        self, identity: SessionIdentity, message: str,
        events: asyncio.Queue,
        product_context: list[dict] | None = None,
    ) -> asyncio.Task:
        """先占用会话，再返回执行任务；HTTP 开流前即可返回 404/429。"""
        entry = await self._ensure_entry(identity)
        self._check_admittable(entry)
        if entry.busy:
            raise SessionBusy("窗口正在处理上一条消息，请稍后再试")
        entry.busy = True
        return asyncio.create_task(self._run_serialized(
            entry, lambda _agent: self._chat_entry(
                entry, message, events, product_context=product_context,
            ),
            reserved=True,
        ))

    async def delete_session(self, identity: SessionIdentity) -> None:
        """关闭当前进程中的 Agent 后删除持久化会话。"""
        self._check_open()
        key = (identity.user_id, identity.session_id)
        entry = self._entries.get(key)
        if entry is not None:
            if entry.busy:
                raise SessionBusy("窗口正在处理上一条消息，请稍后再试")
            entry.closing = True
            await self._close_entry(entry, save=False)
        if self._session_service is not None:
            try:
                await self._session_service.delete_session(
                    identity.user_id, identity.session_id,
                )
            except SessionServiceNotFound as exc:
                raise SessionNotFound("窗口不存在，请先创建") from exc
            return
        if not await asyncio.to_thread(self._verify_owned_file, identity):
            raise SessionNotFound("窗口不存在，请先创建")
        await asyncio.to_thread(
            self._delete_session_file,
            identity,
        )

    def _delete_session_file(self, identity: SessionIdentity) -> None:
        from app.agent.storage import delete_session

        delete_session(
            str(identity.session_path(self._session_dir)), identity=identity,
        )

    async def _run_serialized(
        self, entry: _Entry, operation: Callable[[Any], Any],
        *, reserved: bool = False,
    ) -> Any:
        """chat 专用闸口：busy 同步"检查并占用"，忙时立即拒绝，不排队。

        busy 的检查与设置之间没有 await，原子完成——这比
        ``lock.locked()`` 检查更严格（后者在锁刚释放但有等待者时会漏判，
        导致新请求被迫挂起排队）。窗口锁仍保留，用于与 close/cleanup
        的回收路径互斥；不同窗口的锁互不相同，因此可并行。
        """
        if not reserved:
            if entry.closing:
                raise SessionClosing("窗口正在关闭，请稍后重试")
            if entry.busy:
                raise SessionBusy("窗口正在处理上一条消息，请稍后再试")
            entry.busy = True
        try:
            async with entry.lock:
                # 拿锁期间，窗口可能已经开始关闭。
                if self._closed or entry.closing:
                    raise SessionClosing("窗口正在关闭，请稍后重试")
                try:
                    if entry.agent is None:
                        entry.agent = await asyncio.to_thread(
                            self._factory, entry.identity,
                        )
                        if entry.state is not None:
                            self._restore_agent_state(entry.agent, entry.state)
                    return await operation(entry.agent)
                except Exception:
                    if entry.agent is None:
                        # 初始化失败：旧 Entry 不再接收等待者，新请求可以重建。
                        entry.closing = True
                        self._entries.pop(
                            (entry.identity.user_id, entry.identity.session_id),
                            None,
                        )
                    raise
                finally:
                    entry.last_completed_at = self._clock()
        finally:
            entry.busy = False

    async def _chat_agent(
        self, agent: Any, message: str, events: asyncio.Queue | None = None,
        product_context: list[dict] | None = None,
    ) -> ChatOutcome:
        chat = agent.chat
        supports_events = False
        supports_product_context = False
        if events is not None or product_context:
            try:
                signature = inspect.signature(chat)
                supports_events = "events" in signature.parameters
                supports_product_context = "product_context" in signature.parameters
                has_kwargs = any(item.kind == inspect.Parameter.VAR_KEYWORD
                                 for item in signature.parameters.values())
                supports_events = supports_events or has_kwargs
                supports_product_context = supports_product_context or has_kwargs
            except (TypeError, ValueError):
                supports_events = False
        kwargs = {}
        if events is not None and supports_events:
            kwargs["events"] = events
        if product_context and supports_product_context:
            kwargs["product_context"] = product_context
        reply = await chat(message, **kwargs)
        failures = list(getattr(agent, "last_failures", []))
        failed = any(item.get("affects_answer") for item in failures)
        persisted = not any(item.get("stage") == "persistence" for item in failures)
        warnings = [
            f"{item.get('stage', 'unknown')}_failed".upper()
            for item in failures if not item.get("affects_answer")
        ]
        # 当前方法由 _run_serialized 通过 entry 包装调用；保留这里的返回形状，
        # 数据库快照写入由 _chat_entry 完成。
        return ChatOutcome(
            reply=reply,
            failed=failed,
            persisted=persisted,
            warnings=tuple(warnings),
        )

    async def _chat_entry(
        self, entry: _Entry, message: str, events: asyncio.Queue | None = None,
        product_context: list[dict] | None = None,
    ) -> ChatOutcome:
        outcome = await self._chat_agent(entry.agent, message, events, product_context)
        if self._session_service is None:
            entry.save_failed = not outcome.persisted
            return outcome
        emit(events, "status", stage="save", state="running", message="正在保存会话")
        try:
            entry.state = await self._persist_agent_state(entry)
        except Exception:
            entry.save_failed = True
            emit(events, "status", stage="save", state="failed", message="会话未保存")
            return ChatOutcome(
                reply=outcome.reply,
                failed=outcome.failed,
                persisted=False,
                warnings=outcome.warnings,
            )
        entry.save_failed = False
        emit(events, "status", stage="save", state="done", message="会话已保存")
        return outcome

    async def _persist_agent_state(self, entry: _Entry) -> SessionState:
        if entry.state is None:
            raise SessionNotFound("窗口状态不存在")
        agent = entry.agent
        messages = tuple(_messages_from_agent(agent.raw_messages))
        state = SessionState(
            user_id=entry.state.user_id,
            session_id=entry.state.session_id,
            title=entry.state.title,
            summary=agent.summary,
            short_term_memory=agent.memory_manager.stm_to_dict() or {},
            products=copy.deepcopy(agent.products),
            version=entry.state.version,
            status=entry.state.status,
            created_at=entry.state.created_at,
            updated_at=entry.state.updated_at,
            last_message_at=entry.state.last_message_at,
            messages=messages,
            recommendations=tuple(getattr(agent, "recommendations", ()) or ()),
        )
        return await self._session_service.save_session(
            state, expected_version=entry.state.version,
        )

    @staticmethod
    def _restore_agent_state(agent: Any, state: SessionState) -> None:
        agent.raw_messages = [_agent_message_from_session(item) for item in state.messages]
        agent.summary = state.summary
        agent.products = copy.deepcopy(state.products)
        restore = getattr(agent, "restore_recommendations", None)
        if callable(restore):
            restore(state.recommendations)
        if state.short_term_memory:
            agent.memory_manager.restore_stm(state.short_term_memory)

    # ---- TTL 回收 ----

    async def cleanup(self) -> CleanupReport:
        report = CleanupReport()
        now = self._clock()
        expired = [
            entry for entry in self._entries.values()
            if not entry.closing
            and not entry.busy          # 正在执行的窗口本轮跳过
            and not entry.save_failed   # 持久化失败的窗口暂不回收
            and entry.agent is not None
            and now - entry.last_completed_at >= self._ttl_seconds
        ]
        for entry in expired:
            entry.closing = True

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
        """TTL 回收：只释放 agent 资源，不 save——每轮 chat 结束时
        agent 已自行保存（orchestrator._save_safely），内存中没有未落盘数据。"""
        try:
            await self._close_entry(entry, save=False)
        except Exception as exc:
            entry.closing = False
            return f"{type(exc).__name__}: {exc}"

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
        entries = list(self._entries.values())
        for entry in entries:
            entry.closing = True

        for entry in entries:
            try:
                await asyncio.wait_for(
                    self._close_entry(entry, save=True), timeout=grace_seconds,
                )
            except asyncio.TimeoutError:
                report.timed_out.append(entry.identity.session_id)
                report.unsaved.append(entry.identity.session_id)
                continue
            except Exception as exc:
                report.failed[entry.identity.session_id] = (
                    f"{type(exc).__name__}: {exc}"
                )
                report.unsaved.append(entry.identity.session_id)
                continue
            report.closed.append(entry.identity.session_id)
        return report

    async def _close_entry(self, entry: _Entry, *, save: bool = True) -> None:
        """拿到窗口锁后（按需）保存并关闭，避免和聊天同时操作 Agent。

        save=True 用于服务关闭（防止进程退出前有未保存状态）；
        save=False 用于 TTL 回收（每轮 chat 已自行保存，无需重复）。
        """
        async with entry.lock:
            if entry.agent is not None:
                if save:
                    if self._session_service is None:
                        await asyncio.to_thread(entry.agent.save)
                    elif entry.save_failed:
                        entry.state = await self._persist_agent_state(entry)
                        entry.save_failed = False
                await entry.agent.close()
                entry.agent = None
            key = (entry.identity.user_id, entry.identity.session_id)
            if self._entries.get(key) is entry:
                self._entries.pop(key)

    def session_count(self) -> int:
        return len(self._entries)


def _messages_from_agent(raw_messages: list[dict]) -> list[SessionMessage]:
    """把 OpenAI message dict 转成 JSONB 快照契约。"""
    out: list[SessionMessage] = []
    for raw in raw_messages:
        role = raw.get("role")
        if role not in {"user", "assistant", "tool"}:
            continue
        metadata = {
            key: copy.deepcopy(value)
            for key, value in raw.items()
            if key not in {"role", "content", "tool_call_id"}
        }
        content = raw.get("content")
        out.append(
            SessionMessage(
                role=role,
                content=content if isinstance(content, str) else "",
                tool_call_id=raw.get("tool_call_id"),
                metadata=metadata,
            )
        )
    return out


def _agent_message_from_session(message: SessionMessage) -> dict:
    """把 JSONB 快照消息恢复成 Agent 使用的 OpenAI message dict。"""
    raw = {
        "role": message.role,
        "content": message.content,
        **copy.deepcopy(message.metadata),
    }
    if message.tool_call_id is not None:
        raw["tool_call_id"] = message.tool_call_id
    return raw
