"""离线构建知识库向量索引（Milvus 后端）。

用法：
  python app/scripts/build_kb_index.py

流程：
  1. 扫描 knowledge/ 下的所有 .md 文件，按二级标题切分。
  2. 调用 Embeddings 将每个 chunk 向量化。
  3. MilvusBackend.upsert 全量重建索引
     （uri 指向本地 .db = milvus-lite，指向 http:// = standalone，同一套代码）。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.agent.rag.chunker import chunk_markdown_dir  # noqa: E402
from app.agent.rag.embedder import Embedder  # noqa: E402
from app.agent.rag.milvus_backend import MilvusBackend  # noqa: E402


def main():
    kb_dir = ROOT / settings.kb_dir
    # lite（本地相对路径）按项目根解析；standalone（http://）原样透传
    uri = settings.milvus_uri
    if not uri.startswith("http"):
        uri = str(ROOT / uri)

    if not kb_dir.exists():
        print(f"❌ 知识库目录不存在: {kb_dir}")
        sys.exit(1)

    backend = MilvusBackend(uri=uri, collection_name=settings.milvus_collection)

    print("=" * 60)
    print("  并夕夕 · 知识库索引构建")
    print(f"  后端      : milvus ({'lite' if not uri.startswith('http') else 'standalone'})")
    print(f"  源目录    : {kb_dir}")
    print(f"  索引目标  : {uri} (collection={settings.milvus_collection})")
    print(f"  Embedding : {settings.embedding_model}")
    print("=" * 60)

    print("\n[1/3] 扫描并切分 markdown 文档...")
    chunks = chunk_markdown_dir(kb_dir)
    if not chunks:
        print("❌ 未发现任何文档，请检查 knowledge/ 目录")
        sys.exit(1)

    # 概览块治理（2026-09-14 基线诊断）：每个文档的目录式概览块（#00-00）对任何
    # query 都有词面重叠，会挤占 top3（基线：5/132 槽位、1 次(rank1) 占位），
    # 且对 LLM 回答无增量价值——不进索引。排除概览后每文档首个实义 section 上移。
    before = len(chunks)
    chunks = [c for c in chunks if not c.chunk_id.endswith("#00-00")]
    print(f"   概览块治理：排除 {before - len(chunks)} 个目录式概览块")

    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.doc] = by_doc.get(c.doc, 0) + 1
    for doc, n in by_doc.items():
        print(f"   - {doc}: {n} chunk")
    print(f"   合计 {len(chunks)} 个 chunk")

    print(f"\n[2/3] 调用 {settings.embedding_model} 批量向量化...")
    embedder = Embedder(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )
    vectors = embedder.encode([c.text for c in chunks])
    dim = len(vectors[0]) if vectors else 0
    print(f"   完成，向量维度 = {dim}")

    print("\n[3/3] 写入 milvus 索引...")
    backend.upsert(
        chunks=chunks,
        vectors=vectors,
        embedding_model=settings.embedding_model,
    )
    print(f"   已写入 {backend.size()} 条向量")

    print("\n🎉 索引构建完成。")


if __name__ == "__main__":
    main()
