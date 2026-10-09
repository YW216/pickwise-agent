"""向量化封装：调用 OpenAI Embeddings 接口。

- 支持批量编码（list[str] → list[list[float]]）。
- 可配置 model 与 base_url（与 chat 模型共用一套 OpenAI 客户端配置）。
- 返回原始 list[float]，由调用方决定如何持久化（当前由 MilvusBackend 落库）。
"""

from typing import Iterable

from openai import OpenAI


class Embedder:
    """OpenAI Embeddings 同步封装。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        # 默认值刻意与 settings.embedding_model 保持一致（BAAI/bge-m3）。
        # 原为 text-embedding-3-small，与项目配置脱节：绕过 Settings 直接实例化
        # （如 tests/ 里的 verify 脚本）时会拿到与生产不同的模型，
        # 而 Embedder.retriever 又有「索引模型 vs 当前模型一致性校验」——
        # 不一致时报错要求重建索引，代价远高于默认值本身。
        # 另见 settings.effective_embedding_model：换 provider 时模型名需一并切换
        # （DeepSeek 官方不提供 embedding 服务）。
        model: str = "BAAI/bge-m3",
        batch_size: int = 64,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self._client = OpenAI(
            api_key=api_key, base_url=base_url,
            timeout=timeout, max_retries=max_retries,
        )
        self._model = model
        self._batch_size = batch_size

    @property
    def model(self) -> str:
        return self._model

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        """批量编码，自动按 batch_size 分批请求。"""
        texts = list(texts)
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(item.embedding for item in resp.data)
        return out

    def encode_one(self, text: str) -> list[float]:
        """单条编码，返回原始 list[float]。"""
        return self.encode([text])[0]
