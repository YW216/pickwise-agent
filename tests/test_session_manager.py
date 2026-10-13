"""第三步 SessionManager 验证：并发、容量、TTL、生命周期（多租户隔离和fastapi接入.md §3.9）。

通过公开方法测试；假 factory 计数、事件/屏障控制任务交错、假时钟推进 TTL、
临时文件验证归属与恢复。不靠长 sleep、不调用真实 LLM。所有异步场景用
asyncio.run 包装，不依赖 pytest-asyncio。
"""

import asyncio
import json
import threading
import uuid

import pytest

from app.identity import SessionIdentity
from app.services.session_manager import (
    SessionCapacityFull,
    SessionCreateBusy,
    SessionManager,
    SessionManagerClosed,
    SessionNotFound,
    SessionQueueFull,
)


def uid(name: str) -> str:
    """由名字生成稳定 UUID：测试可读，且满足身份层 UUID 校验。"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


# ---------- 假对象 ----------

class FakeAgent:
    """最小编排器替身：记录调用，支持事件/屏障控制执行交错与注入失败。"""

    def __init__(self, identity: SessionIdentity, barrier: threading.Barrier | None = None,
                 chat_gate: threading.Event | None = None):
        self.identity = identity
        self.barrier = barrier
        self.chat_gate = chat_gate
        self.chat_started = threading.Event()
        self.chat_calls: list[str] = []
        self.chat_total = 0       # 累计执行数（reset 不清），用于顺序断言
        self.chat_error: Exception | None = None
        self.save_error: Exception | None = None
        self.resets = 0
        self.saved = 0
        self.closed = 0
        self.last_failures: list[dict] = []
        self.raw_messages: list[dict] = []
        self.summary = None
        self.products: dict = {}
        self.memory_manager = self  # reset_short_term 直调本对象方法

    def reset_short_term(self) -> None:
        pass

    def chat(self, message: str) -> str:
        self.chat_started.set()
        self.chat_total += 1
        self.chat_calls.append(message)
        if self.barrier is not None:
            self.barrier.wait(timeout=5)  # 串行执行会在此超时破裂 → 测试失败
        if self.chat_gate is not None:
            self.chat_gate.wait(timeout=5)
        if self.chat_error is not None:
            raise self.chat_error
        self.raw_messages.append({"role": "user", "content": message})
        self.raw_messages.append({"role": "assistant", "content": f"reply:{message}"})
        return f"reply:{message}"

    def reset(self) -> None:
        self.resets += 1
        self.chat_calls.clear()
        self.raw_messages.clear()
        self.summary = None
        self.products = {}

    def save(self) -> None:
        self.saved += 1
        if self.save_error is not None:
            raise self.save_error

    def close(self) -> None:
        self.closed += 1


class FakeClock:
    """假单调时钟：测试显式推进 TTL。"""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FactoryRecorder:
    """包装 factory：记录创建的 agent（按 user:session 键索引）与调用次数。"""

    def __init__(self, make_agent=None):
        self.calls: list[SessionIdentity] = []
        self.agents: dict[str, FakeAgent] = {}
        self.error: Exception | None = None
        self._make = make_agent or (lambda identity: FakeAgent(identity))

    def __call__(self, identity: SessionIdentity):
        self.calls.append(identity)
        if self.error is not None:
            raise self.error
        agent = self._make(identity)
        self.agents[f"{identity.user_id}:{identity.session_id}"] = agent
        return agent


# ---------- 工具 ----------

def make_manager(session_dir, clock, factory, **kwargs) -> SessionManager:
    return SessionManager(
        session_dir=session_dir, factory=factory, clock=clock, **kwargs,
    )


async def wait_until(predicate, timeout: float = 3.0) -> None:
    """短轮询等待条件成立（替代长 sleep，不引入固定时序依赖）。"""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("等待条件超时")
        await asyncio.sleep(0.005)


def run(coro):
    return asyncio.run(coro)


# ---------- P3-1：窗口创建、身份定位、按需 factory ----------

def test_create_session_writes_owned_empty_file(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        mgr = make_manager(tmp_path, clock, recorder)
        identity = await mgr.create_session(uid("u1"))
        path = identity.session_path(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["user_id"] == identity.user_id
        assert data["session_id"] == identity.session_id
        assert data["messages"] == []
        assert recorder.calls == []  # 创建窗口不创建编排器

    asyncio.run(main())


def test_unknown_session_cannot_be_claimed(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        mgr = make_manager(tmp_path, clock, recorder)
        stranger = SessionIdentity(
            user_id="11111111-1111-1111-1111-111111111111",
            session_id="22222222-2222-2222-2222-222222222222",
        )
        with pytest.raises(SessionNotFound):
            await mgr.chat(stranger, "hi")
        assert recorder.calls == []  # 未创建的窗口不触发 factory

    asyncio.run(main())


def test_chat_initializes_agent_once_and_reuses(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        mgr = make_manager(tmp_path, clock, recorder)
        identity = await mgr.create_session(uid("u1"))
        first = await mgr.chat(identity, "m1")
        second = await mgr.chat(identity, "m2")
        assert (first.reply, second.reply) == ("reply:m1", "reply:m2")
        assert len(recorder.calls) == 1  # 同窗口多次请求只创建一次
        agent = recorder.agents[f"{identity.user_id}:{identity.session_id}"]
        assert agent.chat_calls == ["m1", "m2"]  # FIFO 顺序执行

    asyncio.run(main())


def test_concurrent_first_chat_creates_single_entry(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        mgr = make_manager(tmp_path, clock, recorder)
        identity = await mgr.create_session(uid("u1"))
        outcomes = await asyncio.gather(
            mgr.chat(identity, "m1"),
            mgr.chat(identity, "m2"),
            mgr.chat(identity, "m3"),
        )
        assert [o.reply for o in outcomes] == ["reply:m1", "reply:m2", "reply:m3"]
        assert len(recorder.calls) == 1
        assert mgr.session_count() == 1

    asyncio.run(main())


def test_different_users_and_windows_get_different_agents(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        mgr = make_manager(tmp_path, clock, recorder)
        user_a_1 = await mgr.create_session(uid("user-a"))
        user_a_2 = await mgr.create_session(uid("user-a"))
        user_b_1 = await mgr.create_session(uid("user-b"))
        await asyncio.gather(
            mgr.chat(user_a_1, "a1"),
            mgr.chat(user_a_2, "a2"),
            mgr.chat(user_b_1, "b1"),
        )
        assert len(recorder.calls) == 3  # 每个窗口一个对象
        assert mgr.session_count() == 3

    asyncio.run(main())


# ---------- P3-2：FIFO、并行、容量 ----------

def test_same_window_chat_reset_serialized_in_order(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        gate = threading.Event()
        recorder._make = lambda identity: FakeAgent(identity, chat_gate=gate)
        mgr = make_manager(tmp_path, clock, recorder)
        identity = await mgr.create_session(uid("u1"))

        first = asyncio.create_task(mgr.chat(identity, "m1"))
        await wait_until(lambda: len(recorder.agents) == 1)  # factory 完成
        agent = next(iter(recorder.agents.values()))
        await wait_until(agent.chat_started.is_set)  # m1 已在执行并阻塞
        second = asyncio.create_task(mgr.chat(identity, "m2"))
        reset_task = asyncio.create_task(mgr.reset(identity))
        third = asyncio.create_task(mgr.chat(identity, "m3"))

        gate.set()  # 放行 m1，其余按排队顺序执行
        await asyncio.gather(first, second, reset_task, third)
        agent = recorder.agents[f"{identity.user_id}:{identity.session_id}"]
        assert agent.chat_total == 3            # m1、m2、m3 都执行过
        assert agent.resets == 1                # reset 发生在 m2 之后（清掉了 m2 记录）
        assert agent.chat_calls == ["m3"]       # m3 在 reset 之后执行

    asyncio.run(main())


def test_different_windows_run_in_parallel(tmp_path):
    async def main():
        clock = FakeClock()
        barrier = threading.Barrier(2)  # 两个线程必须同时到达，串行即超时破裂
        recorder = FactoryRecorder(
            make_agent=lambda identity: FakeAgent(identity, barrier=barrier),
        )
        mgr = make_manager(tmp_path, clock, recorder, max_parallel_jobs=2)
        win1 = await mgr.create_session(uid("u1"))
        win2 = await mgr.create_session(uid("u2"))
        outcomes = await asyncio.gather(
            mgr.chat(win1, "a"), mgr.chat(win2, "b"),
        )
        assert [o.reply for o in outcomes] == ["reply:a", "reply:b"]

    asyncio.run(main())


def test_queue_full_rejected(tmp_path):
    async def main():
        clock = FakeClock()
        gate = threading.Event()
        recorder = FactoryRecorder(
            make_agent=lambda identity: FakeAgent(identity, chat_gate=gate),
        )
        mgr = make_manager(tmp_path, clock, recorder, max_pending_per_session=1)
        identity = await mgr.create_session(uid("u1"))
        running = asyncio.create_task(mgr.chat(identity, "m1"))
        await wait_until(lambda: len(recorder.agents) == 1)  # factory 完成
        agent = next(iter(recorder.agents.values()))
        await wait_until(agent.chat_started.is_set)  # 占用执行位

        queued = asyncio.create_task(mgr.chat(identity, "m2"))  # pending=1
        key = (identity.user_id, identity.session_id)
        await wait_until(lambda: mgr._entries[key].pending == 1)  # m1 执行中不占名额、m2 已入队
        with pytest.raises(SessionQueueFull):
            await mgr.chat(identity, "m3")  # 排队名额已满

        gate.set()
        await running
        await queued

    asyncio.run(main())


def test_capacity_full_when_no_idle_entry(tmp_path):
    async def main():
        clock = FakeClock()
        gate = threading.Event()
        recorder = FactoryRecorder(
            make_agent=lambda identity: FakeAgent(identity, chat_gate=gate),
        )
        mgr = make_manager(tmp_path, clock, recorder, max_sessions=2)
        win1 = await mgr.create_session(uid("u1"))
        win2 = await mgr.create_session(uid("u2"))
        blockers = [asyncio.create_task(mgr.chat(win1, "a")),
                    asyncio.create_task(mgr.chat(win2, "b"))]
        await wait_until(lambda: len(recorder.agents) == 2)
        with pytest.raises(SessionCapacityFull):
            await mgr.chat(await mgr.create_session(uid("u3")), "c")
        gate.set()
        await asyncio.gather(*blockers)

    asyncio.run(main())


def test_capacity_full_reclaims_longest_idle_first(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        mgr = make_manager(tmp_path, clock, recorder, max_sessions=2)
        win1 = await mgr.create_session(uid("u1"))
        win2 = await mgr.create_session(uid("u2"))
        await mgr.chat(win1, "a")
        clock.advance(10)      # win2 完成得更晚 → win1 最久未用
        await mgr.chat(win2, "b")
        old_agent = recorder.agents[f"{win1.user_id}:{win1.session_id}"]

        win3 = await mgr.create_session(uid("u3"))
        await mgr.chat(win3, "c")  # 容量满 → 自动回收最久未用的 win1
        assert old_agent.closed == 1
        assert win1.session_path(tmp_path).exists()  # 只回收内存，不删窗口文件
        assert mgr.session_count() == 2

    asyncio.run(main())


# ---------- P3-3：reset、TTL、close/shutdown ----------

def test_reset_keeps_ownership_and_clears_history(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        mgr = make_manager(tmp_path, clock, recorder)
        identity = await mgr.create_session(uid("u1"))
        await mgr.chat(identity, "m1")
        path = identity.session_path(tmp_path)

        await mgr.reset(identity)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["messages"] == []  # 原子保存空状态
        assert data["user_id"] == identity.user_id  # 归属保留
        agent = recorder.agents[f"{identity.user_id}:{identity.session_id}"]
        assert agent.resets == 1
        again = await mgr.chat(identity, "m2")  # 同一对象继续可用
        assert again.reply == "reply:m2"

    asyncio.run(main())


def test_ttl_cleanup_reclaims_only_expired_idle(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        mgr = make_manager(tmp_path, clock, recorder, ttl_seconds=100)
        win1 = await mgr.create_session(uid("u1"))
        win2 = await mgr.create_session(uid("u2"))
        await mgr.chat(win1, "a")
        clock.advance(99)
        await mgr.chat(win2, "b")
        clock.advance(10)  # win1 距完成 109s（超时），win2 距完成 10s

        report = await mgr.cleanup()
        assert report.reclaimed == [win1.session_id]
        assert report.failed == {}
        agent1 = recorder.agents[f"{win1.user_id}:{win1.session_id}"]
        assert agent1.closed == 1
        assert win1.session_path(tmp_path).exists()  # 文件与 LTM 不动
        assert mgr.session_count() == 1

    asyncio.run(main())


def test_cleanup_keeps_entry_when_save_fails(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        mgr = make_manager(tmp_path, clock, recorder, ttl_seconds=10)
        identity = await mgr.create_session(uid("u1"))
        await mgr.chat(identity, "m1")
        agent = recorder.agents[f"{identity.user_id}:{identity.session_id}"]
        agent.save_error = RuntimeError("disk full")
        clock.advance(100)

        report = await mgr.cleanup()
        assert report.reclaimed == []
        assert "disk full" in report.failed[identity.session_id]
        assert mgr.session_count() == 1  # 保存失败不得当作已回收移除

        agent.save_error = None  # 修复后可再次回收
        report = await mgr.cleanup()
        assert report.reclaimed == [identity.session_id]

    asyncio.run(main())


def test_close_drains_queue_and_closes_once(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        mgr = make_manager(tmp_path, clock, recorder)
        identity = await mgr.create_session(uid("u1"))
        outcome_task = asyncio.create_task(mgr.chat(identity, "m1"))
        await wait_until(lambda: len(recorder.agents) == 1)  # factory 完成
        agent = recorder.agents[f"{identity.user_id}:{identity.session_id}"]
        await wait_until(agent.chat_started.is_set)

        close_task = asyncio.create_task(mgr.close(grace_seconds=5))
        outcome = await outcome_task  # 已接纳任务在关闭前完成
        assert outcome.reply == "reply:m1"
        report = await close_task

        assert report.closed == [identity.session_id]
        assert agent.closed == 1  # 资源只关闭一次
        with pytest.raises(SessionManagerClosed):
            await mgr.chat(identity, "m2")  # 关闭后拒绝新任务

    asyncio.run(main())


def test_close_reports_timed_out_running_window(tmp_path):
    async def main():
        clock = FakeClock()
        gate = threading.Event()
        recorder = FactoryRecorder(
            make_agent=lambda identity: FakeAgent(identity, chat_gate=gate),
        )
        mgr = make_manager(tmp_path, clock, recorder)
        identity = await mgr.create_session(uid("u1"))
        running = asyncio.create_task(mgr.chat(identity, "m1"))
        await wait_until(lambda: len(recorder.agents) == 1)  # factory 完成
        agent = recorder.agents[f"{identity.user_id}:{identity.session_id}"]
        await wait_until(agent.chat_started.is_set)

        report = await mgr.close(grace_seconds=0.1)  # 宽限内跑不完
        assert report.timed_out == [identity.session_id]
        assert report.unsaved == [identity.session_id]
        assert agent.closed == 0  # 未完成窗口不做半吊子回收

        gate.set()  # 线程不可强杀：放行后自然跑完
        outcome = await running  # 等待协程不被关闭中断，最终拿到执行结果
        assert outcome.reply == "reply:m1"

    asyncio.run(main())


# ---------- P3-4：取消与竞态 ----------

def test_cancelled_pending_job_is_skipped(tmp_path):
    async def main():
        clock = FakeClock()
        gate = threading.Event()
        recorder = FactoryRecorder(
            make_agent=lambda identity: FakeAgent(identity, chat_gate=gate),
        )
        mgr = make_manager(tmp_path, clock, recorder)
        identity = await mgr.create_session(uid("u1"))
        running = asyncio.create_task(mgr.chat(identity, "m1"))
        await wait_until(lambda: len(recorder.agents) == 1)  # factory 完成
        agent = recorder.agents[f"{identity.user_id}:{identity.session_id}"]
        await wait_until(agent.chat_started.is_set)

        queued = asyncio.create_task(mgr.chat(identity, "m2"))  # 排队中
        key = (identity.user_id, identity.session_id)
        await wait_until(lambda: mgr._entries[key].pending == 1)  # m1 已开始（释放名额）、m2 已入队
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued

        gate.set()
        outcome = await running
        assert outcome.reply == "reply:m1"
        assert agent.chat_calls == ["m1"]  # 排队中的 m2 被跳过，未执行

    asyncio.run(main())


def test_cancelling_waiting_caller_does_not_stop_execution(tmp_path):
    async def main():
        clock = FakeClock()
        gate = threading.Event()
        recorder = FactoryRecorder(
            make_agent=lambda identity: FakeAgent(identity, chat_gate=gate),
        )
        mgr = make_manager(tmp_path, clock, recorder)
        identity = await mgr.create_session(uid("u1"))
        caller = asyncio.create_task(mgr.chat(identity, "m1"))
        await wait_until(lambda: len(recorder.agents) == 1)  # factory 完成
        agent = recorder.agents[f"{identity.user_id}:{identity.session_id}"]
        await wait_until(agent.chat_started.is_set)  # 已开始执行

        caller.cancel()  # 模拟客户端断开：不中断已开始的执行
        with pytest.raises(asyncio.CancelledError):
            await caller
        gate.set()
        await wait_until(lambda: len(agent.chat_calls) == 1)  # 执行自然完成

    asyncio.run(main())


def test_create_busy_when_pending_limit_reached(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        mgr = make_manager(tmp_path, clock, recorder, max_pending_creates=0)
        with pytest.raises(SessionCreateBusy):
            await mgr.create_session(uid("u1"))  # 上限为 0：立即拒绝（边界配置验证）

    asyncio.run(main())


def test_factory_failure_removes_entry_and_allows_retry(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        recorder.error = RuntimeError("boom")
        mgr = make_manager(tmp_path, clock, recorder)
        identity = await mgr.create_session(uid("u1"))

        with pytest.raises(RuntimeError, match="boom"):
            await mgr.chat(identity, "m1")
        assert mgr.session_count() == 0  # Entry 已移除

        recorder.error = None  # 稍后重试：重新初始化
        outcome = await mgr.chat(identity, "m1")
        assert outcome.reply == "reply:m1"
        assert len(recorder.calls) == 2

    asyncio.run(main())


def test_closed_manager_rejects_everything(tmp_path):
    async def main():
        clock = FakeClock()
        recorder = FactoryRecorder()
        mgr = make_manager(tmp_path, clock, recorder)
        identity = await mgr.create_session(uid("u1"))
        await mgr.close()
        with pytest.raises(SessionManagerClosed):
            await mgr.create_session(uid("u1"))
        with pytest.raises(SessionManagerClosed):
            await mgr.chat(identity, "m1")

    asyncio.run(main())
