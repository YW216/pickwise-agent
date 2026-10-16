"""Reranker 封装：cross-encoder 精排（硅基流动 /v1/rerank）。

召回（hybrid 粗排）后对候选窗口做 query-doc 精排，提升 top_k 内的
相关密度（Prec）与头部排名（MRR）。API 非 OpenAI 标准（Jina 风格），
直接 httpx POST；配置回退链与 embedding 相同（同源硅基流动）。
"""

from __future__ import annotations

import httpx


class Reranker:
    """rerank API 同步封装：query + 候选文本 → 按相关度重排的下标列表。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "BAAI/bge-reranker-v2-m3",
        *,
        timeout: float = 30.0,
    ):
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/rerank"  # 已含 /v1 前缀
        self._model = model
        self._timeout = timeout

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[int]:
        """返回按相关度降序的**原列表下标**（截取前 top_n 个）。

        调用方持有原始候选列表，按下标取回自己的对象——本模块只做排序。
        失败抛异常，由调用方决定降级（如跳过 rerank 用原始序）。
        """
        if not documents:
            return []
        resp = httpx.post(
            self._url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return [item["index"] for item in resp.json()["results"]]
