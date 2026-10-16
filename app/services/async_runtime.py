"""异步基础层：统一异步依赖入口（异步改造阶段一）。

本模块只建立基础设施，不改变任何业务行为——现有同步调用方全部不动，
异步客户端与连接池由应用 lifespan 创建/释放，供后续阶段逐步切换：

- AsyncModelClient：langfuse 插桩的 AsyncOpenAI（与同步 client 同源观测），
  自带 LLM 并发闸门（asyncio.Semaphore，替代被删线程池的限并发职责）；
- AsyncDatabasePool：psycopg 异步连接池（阶段五数据访问切换时使用）；

同步回退保证：不依赖本模块的任何代码路径照常工作（验收标准 §阶段一）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from langfuse.openai import AsyncOpenAI

from app.config.settings import settings


@dataclass(frozen=True)
class RuntimeConfig:
    """异步基础设施的启动参数（全部来自 settings，测试可覆盖）。"""

    openai_api_key: str
    openai_base_url: str
    openai_timeout: float
    openai_max_retries: int

    @classmethod
    def from_settings(cls) -> "RuntimeConfig":
        return cls(
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
            openai_timeout=settings.openai_timeout,
            openai_max_retries=settings.openai_max_retries,
        )


class AsyncRuntime:
    """应用级异步资源容器：模型客户端 + PG 连接池 + 并发闸门。

    由 create_app 的 lifespan 创建并保存在 app.state，退出时 aclose()；
    每个实例自持资源，多实例（测试）天然隔离。
    """

    def __init__(self, config: RuntimeConfig | None = None):
        self._config = config or RuntimeConfig.from_settings()
        self._model_client: AsyncOpenAI | None = None
        self._closed = False

    # ---- 生命周期 ----

    async def open(self) -> None:
        """创建模型客户端（lifespan 启动时调用一次）。

        PG 异步池由 ``app.db.connection`` 按需惰性创建（阶段五起），
        不再挂在本容器上——工具层拿不到 app.state，模块级池是唯一可行归属。
        """
        if self._model_client is not None:
            raise RuntimeError("AsyncRuntime 已经打开")
        self._model_client = AsyncOpenAI(
            api_key=self._config.openai_api_key,
            base_url=self._config.openai_base_url,
            timeout=self._config.openai_timeout,
            max_retries=self._config.openai_max_retries,
        )

    async def aclose(self) -> None:
        """释放资源（lifespan 退出时调用一次；重复调用安全）。"""
        if self._closed:
            return
        self._closed = True
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


