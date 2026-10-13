"""SessionManager：进程内多窗口会话管理（第三步，多租户隔离和fastapi接入.md §3）。

一个 backend 进程一个 SessionManager；每个 (user_id, session_id) 窗口对应
一个 Entry：身份、懒创建的编排器、FIFO 任务队列、唯一消费者、状态与最近
完成时间。窗口操作（chat/reset）全部经队列由消费者串行执行，同窗口严格
不交错、跨窗口并行；阻塞工作跑在 SessionManager 自己的线程池里，
admission semaphore 限制同时在跑的任务数。

职责边界：
- 只管理内存对象与窗口文件，不删除窗口文件和用户长期记忆（TTL 回收
  仅回收内存对象）；
- 不暴露 Entry 与编排器对象——调用方只能走公开方法，杜绝绕过队列
  直接操作运行对象；
- 不做请求幂等/任务查询（后续增量），HTTP 断开不中断已开始的执行。
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from app.agent.storage import load_session, save_session
from app.identity import SessionIdentity, SessionOwnershipError, validate_user_id


# ---------- 错误类型（code 供 HTTP 适配层映射状态码） ----------

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


# ---------- 结果与报告 ----------

@dataclass(frozen=True)
class ChatOutcome:
    """单次 chat 的结果快照：消费者内生成，之后不再随窗口状态变化。"""

    reply: str
    failed: bool                 # 是否发生影响答案的失败
    persisted: bool              # 本轮是否成功落盘
    warnings: tuple[str, ...] = ()


@dataclass
class CleanupReport:
    """一次 cleanup 的结果：回收成功的窗口与保留（失败）的窗口。"""

    reclaimed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


@dataclass
class ShutdownReport:
    """close() 的结果：正常关闭、保存失败、宽限超时的窗口清单。"""

    closed: list[str] = field(default_factory=list)
    unsaved: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    timed_out: list[str] = field(default_factory=list)


# ---------- 内部结构 ----------

class SessionState(Enum):
    INITIALIZING = auto()   # factory 执行中（编排器尚不可用，操作只能排队）
    IDLE = auto()           # 无排队、无执行
    WAITING = auto()        # 队列非空，消费者尚未取完
    RUNNING = auto()        # 正在执行一个操作
    CLOSING = auto()        # 回收中：拒绝接纳新任务
    CLOSED = auto()         # 已关闭


@dataclass
class _Job:
    """一条排队操作；cancelled 只在事件循环线程内读写，无锁安全。"""

    kind: str                       # "chat" | "reset"
    message: str | None
    future: asyncio.Future
    cancelled: bool = False


_STOP = object()  # shutdown 哨兵：消费者收到即退出


class _Entry:
    """窗口运行记录：队列、消费者与编排器随 Entry 一起管理。"""

    __slots__ = ("identity", "state", "agent", "queue", "consumer",
                 "pending", "last_completed_at", "init_error")

    def __init__(self, identity: SessionIdentity, now: float):
        self.identity = identity
        self.state = SessionState.INITIALIZING
        self.agent: Any = None                 # factory 产物，初始化完成前为 None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.consumer: asyncio.Task | None = None
        self.pending = 0                       # 已接纳未完成数（含等待线程名额的）
        self.last_completed_at = now
        self.init_error: BaseException | None = None


# ---------- SessionManager ----------

class SessionManager:
    """管理 (user, window) → 编排器 的并发、容量与生命周期。

    Args:
        session_dir：Web 会话文件根目录（窗口文件写在 users/<uid>/<sid>.json）
        factory：identity → 运行对象（生产为 MultiAgentOrchestrator，测试可注入）
        clock：单调时钟（默认 time.monotonic，测试可注入假时钟）
        executor：可选外接线程池；缺省自建 max_parallel_jobs 规模并在 close 关闭
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
        executor: ThreadPoolExecutor | None = None,
    ):
        self._session_dir = Path(session_dir)
        self._factory = factory
        self._clock = clock or time.monotonic
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._max_pending = max_pending_per_session
        if executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=max_parallel_jobs, thread_name_prefix="session-worker",
            )
            self._owns_executor = True
        else:
            self._executor = executor
            self._owns_executor = False
        self._admission = asyncio.Semaphore(max_parallel_jobs)
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._registry_lock = asyncio.Lock()   # 保护 entries/容量/pending 的检查与修改
        self._create_pending = 0               # 全局待执行创建数（含等待线程的）
        self._max_pending_creates = max_pending_creates
        self._closed = False
        self._cleanup_task: asyncio.Task | None = None

    # ---- 创建窗口（3.3） ----

    async def create_session(self, user_id: str) -> SessionIdentity:
        """显式创建窗口：生成编号并在工作线程写入带归属的空会话文件。

        不调用 LLM、不创建编排器；编号撞已有文件则重新生成（有界重试）。
        全局待执行创建数超限时返回 SESSION_CREATE_BUSY，避免创建请求
        无限等待线程名额。
        """
        self._check_open()
        validate_user_id(user_id)
        async with self._registry_lock:
            if self._create_pending >= self._max_pending_creates:
                raise SessionCreateBusy("窗口创建请求过多，请稍后重试")
            self._create_pending += 1
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self._create_session_file, user_id,
            )
        finally:
            async with self._registry_lock:
                self._create_pending -= 1

    def _create_session_file(self, user_id: str) -> SessionIdentity:
        for _ in range(5):  # UUID 撞文件概率≈0，5 次重试仅作防御
            identity = SessionIdentity(user_id=user_id, session_id=str(uuid4()))
            path = identity.session_path(self._session_dir)
            if path.exists():
                continue
            try:
                save_session(str(path), [], None, identity=identity)
            except SessionOwnershipError:
                continue  # 极小概率的并发同名：换一个编号重来
            return identity
        raise SessionManagerError("无法分配新的窗口编号（连续冲突）")

    # ---- chat / reset：定位窗口、入队（3.3/3.4） ----

    async def chat(self, identity: SessionIdentity, message: str) -> ChatOutcome:
        """向窗口投递一条聊天消息；返回结果快照（客户端断开不中断执行）。"""
        job = await self._submit(identity, kind="chat", message=message)
        return await self._await_job(job)

    async def reset(self, identity: SessionIdentity) -> None:
        """重置窗口（走队列与 chat 串行）；只清历史/STM/商品提及，不动 LTM。"""
        job = await self._submit(identity, kind="reset", message=None)
        await self._await_job(job)

    async def _submit(self, identity: SessionIdentity, *, kind: str,
                      message: str | None) -> _Job:
        self._check_open()
        entry = await self._ensure_entry(identity)
        loop = asyncio.get_running_loop()
        job = _Job(kind=kind, message=message, future=loop.create_future())
        async with self._registry_lock:
            self._check_admittable(entry)
            if entry.state is SessionState.CLOSED and entry.init_error is not None:
                raise entry.init_error  # 初始化失败的窗口：原样回错，可稍后重试
            if entry.pending >= self._max_pending:
                raise SessionQueueFull(f"窗口排队已达上限（{self._max_pending}）")
            entry.pending += 1
            if entry.state is not SessionState.INITIALIZING:
                entry.state = SessionState.WAITING
            entry.queue.put_nowait(job)
        return job

    async def _await_job(self, job: _Job):
        """等待结果；本协程被取消（客户端断开）时标记排队任务可跳过。

        已开始的执行不受影响（3.5）：取消只对「尚未开始」的任务生效；
        future 的结果/异常始终由消费者设置，不产生无人接收的异常。
        """
        try:
            return await job.future
        except asyncio.CancelledError:
            job.cancelled = True  # 事件循环内单标志，消费者取出时检查
            raise

    # ---- Entry 定位与创建（3.3：创建窗口与创建运行对象是两步） ----

    async def _ensure_entry(self, identity: SessionIdentity, *,
                            allow_reclaim: bool = True) -> _Entry:
        key = (identity.user_id, identity.session_id)
        candidates = None
        async with self._registry_lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._check_admittable(entry)
                return entry
            if len(self._entries) >= self._max_sessions:
                # 容量已满：锁内只做内存标记，回收（文件 I/O）在锁外进行
                candidates = self._mark_reclaimable_locked()

        if candidates:
            for entry in candidates:
                await self._reclaim_one(entry)
            # 回收后重试一次；再满则直接 CAPACITY_FULL（避免失败循环）
            return await self._ensure_entry(identity, allow_reclaim=False)

        # 锁外做文件 I/O：核对文件存在与归属（不允许未创建的窗口被认领）
        loop = asyncio.get_running_loop()
        found = await loop.run_in_executor(
            self._executor, self._verify_owned_file, identity,
        )
        if not found:
            raise SessionNotFound("窗口不存在，请先创建")

        async with self._registry_lock:
            entry = self._entries.get(key)  # double-check：并发首访只建一次
            if entry is not None:
                self._check_admittable(entry)
                return entry
            if len(self._entries) >= self._max_sessions:
                raise SessionCapacityFull(
                    f"运行窗口数已达上限（{self._max_sessions}）且无空闲可回收")
            entry = _Entry(identity, now=self._clock())
            self._entries[key] = entry
            entry.consumer = asyncio.create_task(self._consume(entry))
            return entry

    def _verify_owned_file(self, identity: SessionIdentity) -> bool:
        """核对窗口文件存在且归属匹配；旧 CLI 文件（无归属）一律拒绝。"""
        loaded = load_session(
            str(identity.session_path(self._session_dir)), identity=identity,
        )
        return loaded is not None

    def _check_open(self) -> None:
        if self._closed:
            raise SessionManagerClosed("SessionManager 已关闭")

    @staticmethod
    def _check_admittable(entry: _Entry) -> None:
        if entry.state in (SessionState.CLOSING, SessionState.CLOSED):
            raise SessionClosing("窗口正在关闭，请稍后重试")

    def _mark_reclaimable_locked(self) -> list[_Entry]:
        """容量已满：标记最久未用的空闲窗口为回收候选（锁内只改内存）。

        只回收一个（腾出名额即可）；无空闲可回收时抛 SESSION_CAPACITY_FULL。
        """
        idle = sorted(
            (e for e in self._entries.values()
             if e.state is SessionState.IDLE and e.pending == 0),
            key=lambda e: e.last_completed_at,
        )
        if not idle:
            raise SessionCapacityFull(
                f"运行窗口数已达上限（{self._max_sessions}）且无空闲可回收")
        victim = idle[0]
        victim.state = SessionState.CLOSING
        return [victim]

    # ---- 消费者（3.4：每窗口一个 FIFO 消费者） ----

    async def _consume(self, entry: _Entry) -> None:
        """窗口唯一消费者：先初始化编排器，再按队列顺序执行操作。

        初始化与每个操作都在工作线程执行；同一窗口任一时刻至多一个
        操作在跑。消费者属于 SessionManager，不属于任何请求协程。
        """
        key = (entry.identity.user_id, entry.identity.session_id)
        loop = asyncio.get_running_loop()
        try:
            entry.agent = await loop.run_in_executor(
                self._executor, self._factory, entry.identity,
            )
        except Exception as exc:  # noqa: BLE001 —— 初始化失败：清理并移除，可稍后重试
            await self._fail_initialization(entry, exc)
            return
        entry.state = SessionState.IDLE
        entry.last_completed_at = self._clock()

        while True:
            job = await entry.queue.get()
            if job is _STOP:
                return
            if job.cancelled:
                self._release_pending(entry, job, result=None)
                continue
            async with self._admission:  # 异步等线程名额，不占用工作线程
                if job.cancelled:        # 等名额期间被取消：提交前再检查
                    self._release_pending(entry, job, result=None)
                    continue
                # 开始执行：不再占「未开始」排队名额（3.4 口径）
                entry.pending -= 1
                entry.queue.task_done()
                entry.state = SessionState.RUNNING
                try:
                    result = await loop.run_in_executor(
                        self._executor, self._run_job_sync, entry, job,
                    )
                except Exception as exc:  # noqa: BLE001 —— 执行器层兜底
                    job.future.set_exception(exc)
                    result = None
                finally:
                    entry.state = (
                        SessionState.WAITING if not entry.queue.empty()
                        else SessionState.IDLE
                    )
                    entry.last_completed_at = self._clock()
                if not job.future.done():
                    job.future.set_result(result)

    def _run_job_sync(self, entry: _Entry, job: _Job):
        """工作线程内执行单个操作并生成结果快照（3.2：不得延后读取状态）。"""
        if job.kind == "chat":
            agent = entry.agent
            reply = agent.chat(job.message)
            failures = list(getattr(agent, "last_failures", []))
            failed = any(f.get("affects_answer") for f in failures)
            persisted = not any(f.get("stage") == "persistence" for f in failures)
            # 警告给稳定 code（HTTP 层直接透传，§4.5）；诊断细节留服务端日志
            warnings = tuple(
                f"{f.get('stage', 'unknown')}_failed".upper()
                for f in failures if not f.get("affects_answer")
            )
            return ChatOutcome(reply=reply, failed=failed,
                               persisted=persisted, warnings=warnings)
        entry.agent.reset()  # reset：编排器统一「持久化空状态 → 清内存」语义
        return None

    def _release_pending(self, entry: _Entry, job: _Job, *, result) -> None:
        """收尾一条未开始的任务（取消跳过）：释放名额并给占位结果。"""
        entry.pending -= 1
        entry.queue.task_done()
        if result is None:  # 取消跳过的任务：给占位结果，避免 future 悬挂
            result = ChatOutcome(reply="", failed=True, persisted=False,
                                 warnings=("请求已取消",))
        if not job.future.done():
            job.future.set_result(result)

    async def _fail_initialization(self, entry: _Entry, exc: Exception) -> None:
        """factory 失败：移除 Entry、给排队任务回错；可稍后重试（3.7）。"""
        entry.state = SessionState.CLOSED
        entry.init_error = exc
        key = (entry.identity.user_id, entry.identity.session_id)
        async with self._registry_lock:
            self._entries.pop(key, None)
        while not entry.queue.empty():
            job = entry.queue.get_nowait()
            if job is not _STOP and not job.future.done():
                job.future.set_exception(exc)
            entry.pending -= 1

    # ---- TTL 与容量回收（3.6） ----

    async def cleanup(self) -> CleanupReport:
        """回收超过 TTL 的空闲窗口：只回收内存对象，不删窗口文件。"""
        report = CleanupReport()
        now = self._clock()
        async with self._registry_lock:
            expired = [
                entry for entry in self._entries.values()
                if entry.state is SessionState.IDLE
                and entry.pending == 0
                and now - entry.last_completed_at >= self._ttl_seconds
            ]
            for entry in expired:
                entry.state = SessionState.CLOSING  # 停止接纳，防止回收中复用
        for entry in expired:
            key = entry.identity.session_id
            error = await self._reclaim_one(entry)
            if error is None:
                report.reclaimed.append(key)
            else:
                report.failed[key] = error  # 保留对象（state 已回 IDLE），稍后重试
        return report

    async def start_cleanup_loop(self, interval_seconds: float = 60.0) -> asyncio.Task:
        """启动周期 cleanup（lifespan 用）；close 时自动取消。"""
        self._check_open()

        async def _loop():
            while True:
                await asyncio.sleep(interval_seconds)
                try:
                    await self.cleanup()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 —— 周期任务失败不影响服务
                    pass

        self._cleanup_task = asyncio.create_task(_loop())
        return self._cleanup_task

    async def _reclaim_one(self, entry: _Entry) -> str | None:
        """单个窗口的回收流程；返回 None 表示成功，错误字符串表示保留。"""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._reclaim_sync, entry)
        except Exception as exc:  # noqa: BLE001 —— 保存/关闭失败：保留对象稍后重试
            entry.state = SessionState.IDLE
            return f"{type(exc).__name__}: {exc}"
        if entry.consumer is not None:
            entry.consumer.cancel()  # 队列已空、无执行中任务，安全取消
        async with self._registry_lock:
            self._entries.pop((entry.identity.user_id, entry.identity.session_id), None)
        return None

    def _reclaim_sync(self, entry: _Entry) -> None:
        """工作线程内：save → close；close 只执行一次（state 幂等闸门）。"""
        if entry.state is SessionState.CLOSED:
            return
        entry.agent.save()
        entry.agent.close()
        entry.state = SessionState.CLOSED

    # ---- 退出（3.8） ----

    async def close(self, grace_seconds: float = 30.0) -> ShutdownReport:
        """关闭 SessionManager：停接纳 → 排空队列 → 逐窗口保存关闭 → 关线程池。

        宽限时间内等不完的窗口记入 timed_out（Python 线程不可强杀，
        只保证已写入的文件）；单窗口失败不影响其余窗口。
        """
        if self._closed:
            raise SessionManagerClosed("SessionManager 已关闭")
        self._closed = True
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None

        report = ShutdownReport()
        entries = list(self._entries.values())
        for entry in entries:
            entry.state = SessionState.CLOSING   # 停止接纳该窗口的新任务
            entry.queue.put_nowait(_STOP)        # 唤醒空队列上挂起的消费者
        consumers = [e.consumer for e in entries if e.consumer is not None]
        if consumers:
            await asyncio.wait(consumers, timeout=grace_seconds)
        for entry in entries:
            if entry.consumer is not None and not entry.consumer.done():
                report.timed_out.append(entry.identity.session_id)

        loop = asyncio.get_running_loop()
        for entry in entries:
            key = entry.identity.session_id
            if entry.state is SessionState.CLOSED:
                continue
            if entry.consumer is not None and not entry.consumer.done():
                report.unsaved.append(key)  # 仍在执行：不安全回收，如实报告
                continue
            try:
                await loop.run_in_executor(self._executor, self._reclaim_sync, entry)
            except Exception as exc:  # noqa: BLE001 —— 单窗口失败继续其余
                report.failed[key] = f"{type(exc).__name__}: {exc}"
                report.unsaved.append(key)
                continue
            if entry.consumer is not None:
                entry.consumer.cancel()
            async with self._registry_lock:
                self._entries.pop((entry.identity.user_id, key), None)
            report.closed.append(key)

        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        return report

    # ---- 诊断 ----

    def session_count(self) -> int:
        """当前持有的 Entry 数（含 INITIALIZING/CLOSING）。"""
        return len(self._entries)
