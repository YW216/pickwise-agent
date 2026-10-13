"""异步基础层：统一异步依赖入口（异步改造阶段一）。

本模块只建立基础设施，不改变任何业务行为——现有同步调用方全部不动，
异步客户端与连接池由应用 lifespan 创建/释放，供后续阶段逐步切换：

- AsyncModelClient：langfuse 插桩的 AsyncOpenAI（与同步 client 同源观测），
  自带 LLM 并发闸门（asyncio.Semaphore，替代被删线程池的限并发职责）；
- AsyncDatabasePool：psycopg 异步连接池（阶段五数据访问切换时使用）；
- run_sync：同步调用的过渡包装（阶段二/三混搭期统一 to_thread 出口）。

同步回退保证：不依赖本模块的任何代码路径照常工作（验收标准 §阶段一）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from langfuse.openai import AsyncOpenAI
from psycopg_pool import AsyncConnectionPool

from app.config.settings import settings


@dataclass(frozen=True)
class RuntimeConfig:
    """异步基础设施的启动参数（全部来自 settings，测试可覆盖）。"""

    openai_api_key: str
    openai_base_url: str
    openai_timeout: float
    openai_max_retries: int
    llm_max_concurrency: int
    database_url: str
    db_pool_min_size: int
    db_pool_max_size: int

    @classmethod
    def from_settings(cls) -> "RuntimeConfig":
        return cls(
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
            openai_timeout=settings.openai_timeout,
            openai_max_retries=settings.openai_max_retries,
            llm_max_concurrency=settings.llm_max_concurrency,
            database_url=settings.database_url,
            db_pool_min_size=settings.db_pool_min_size,
            db_pool_max_size=settings.db_pool_max_size,
        )


class LlmConcurrencyExceeded(Exception):
    """LLM 并发闸门拒绝（保留给未来「满载直接 429」的选项；当前默认排队）。"""


class AsyncRuntime:
    """应用级异步资源容器：模型客户端 + PG 连接池 + 并发闸门。

    由 create_app 的 lifespan 创建并保存在 app.state，退出时 aclose()；
    每个实例自持资源，多实例（测试）天然隔离。
    """

    def __init__(self, config: RuntimeConfig | None = None):
        self._config = config or RuntimeConfig.from_settings()
        self._model_client: AsyncOpenAI | None = None
        self._pool: AsyncConnectionPool | None = None
        self._llm_slots = asyncio.Semaphore(self._config.llm_max_concurrency)
        self._closed = False

    # ---- 生命周期 ----

    async def open(self) -> None:
        """创建模型客户端与 PG 连接池（lifespan 启动时调用一次）。"""
        if self._model_client is not None or self._pool is not None:
            raise RuntimeError("AsyncRuntime 已经打开")
        self._model_client = AsyncOpenAI(
            api_key=self._config.openai_api_key,
            base_url=self._config.openai_base_url,
            timeout=self._config.openai_timeout,
            max_retries=self._config.openai_max_retries,
        )
        # open=False + 后台初始化：PG 临时不可达不阻塞应用启动（§4.6），
        # 依赖状态由 /api/ready 暴露；首次 connection() 时才真正建连
        self._pool = AsyncConnectionPool(
            conninfo=self._config.database_url,
            min_size=self._config.db_pool_min_size,
            max_size=self._config.db_pool_max_size,
            open=False,
        )
        await self._pool.open()  # wait=False：后台填充连接，失败自动重试

    async def aclose(self) -> None:
        """释放资源（lifespan 退出时调用一次；重复调用安全）。"""
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        if self._model_client is not None:
            try:
                await self._model_client.close()
            except Exception:  # noqa: BLE001 —— 退出清理尽力而为
                pass
            self._model_client = None

    # ---- 访问入口 ----

    @property
    def model_client(self) -> AsyncOpenAI | None:
        """异步模型客户端；未打开/已关闭时为 None（业务在 open 后才取用）。"""
        return self._model_client

    @property
    def pool(self) -> AsyncConnectionPool | None:
        """异步 PG 连接池；未打开/已关闭时为 None。"""
        return self._pool

    @property
    def llm_concurrency(self) -> int:
        """当前可用 LLM 名额（诊断用）。"""
        return self._llm_slots._value

    @property
    def llm_gate(self) -> asyncio.Semaphore:
        """LLM 并发闸门（orchestrator 的 GatedAsyncClient 使用）。"""
        return self._llm_slots

    def acquire_llm_slot(self) -> "_LlmSlot":
        """获取一个 LLM 并发名额句柄（模型调用前 acquire，结束后释放）。

        用法：``async with runtime.acquire_llm_slot(): await client...``
        名额耗尽时协程排队等待（不占线程、不拒绝请求）。
        """
        return _LlmSlot(self._llm_slots)


class _LlmSlot:
    """async with 风格的并发名额句柄。"""

    def __init__(self, semaphore: asyncio.Semaphore):
        self._semaphore = semaphore

    async def __aenter__(self):
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, *exc_info):
        self._semaphore.release()
        return False


async def run_sync(fn, *args, **kwargs):
    """同步调用的过渡包装：混搭期统一 to_thread 出口（阶段二/三使用）。

    所有「事件循环内需要调用同步代码」的地方一律走这里，禁止散落
    裸 to_thread——便于阶段八统一盘点与清理。
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


class GatedAsyncClient:
    """AsyncOpenAI 的并发闸门包装（阶段二的最小侵入关键）。

    Router / SubAgent / compaction / extraction 只调
    ``client.chat.completions.create``——把传入的 client 换成这个包装后，
    调用点零改动，限流在 client 层统一生效（每次 create 过 Semaphore）。
    其余属性（model / close / usage 统计等）透传给内部真实客户端。
    """

    def __init__(self, inner, semaphore: asyncio.Semaphore):
        self._inner = inner
        self._semaphore = semaphore
        # 测试替身可能只有 close（无 chat 链）：缺失时走 __getattr__ 透传，
        # 访问时才报错，保证构造永不因替身形状失败
        if hasattr(inner, "chat"):
            self.chat = _GatedChat(inner.chat, semaphore)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _GatedChat:
    def __init__(self, inner_chat, semaphore: asyncio.Semaphore):
        self.completions = _GatedCompletions(inner_chat.completions, semaphore)


class _GatedCompletions:
    def __init__(self, inner_completions, semaphore: asyncio.Semaphore):
        self._inner = inner_completions
        self._semaphore = semaphore

    async def create(self, **kwargs):
        async with self._semaphore:  # 名额耗尽时排队，不占线程
            return await self._inner.create(**kwargs)
