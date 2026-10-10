"""商品索引增量同步核心（service 层）：PG 真值 → Milvus 检索面。

职责边界（develop_docs/rag增量更新.md 第五节）：
- 本模块只实现 sync 核心逻辑（diff / 契约校验 / 局部写入 / 状态提交），
  不含 CLI / worker / FastAPI 任何外壳——三壳共用，换壳不换逻辑
- SQL 归 catalog_repo；collection 的建库（drop/create）归 build_product_kb
  的 rebuild 模式——本模块对 product_kb 只做 upsert / delete / query，绝不 drop

幂等的三个支柱：
1. 稳定主键：product_id 即 Milvus 主键，upsert / delete 按主键天然幂等
2. 状态最后提交：Milvus 写完并抽查通过，才写回 card_hash / indexed_at；
   中途失败不提交，下一轮 diff 自然重做
3. 乐观锁提交：标记带 WHERE updated_at = 快照值，同步期间又被改的行
   不标记（防"改了却被标已索引"的静默丢失），下一轮重做

diff 单 hash 设计：
- card_hash 只判"向量是否需要重算"；行数据（price / category）永远取
  PG 真值整行 upsert——一条规则，没有"哪个字段变了"的判断分支
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from app.agent.rag.embedder import Embedder
from app.agent.rag.milvus_utils import ensure_reachable
from app.config.settings import settings

__all__ = [
    "TEMPLATE_VERSION",
    "SyncPlan",
    "SyncStats",
    "build_card_text",
    "card_hash",
    "build_contract",
    "parse_contract",
    "plan_sync",
    "sync",
]

# 卡片模板版本：任何影响 text 的改动必须递增并触发全量重建
# （build_product_kb 与本模块共用，契约指纹携带此版本参与校验）
TEMPLATE_VERSION = "product_card_template_v2"

# ---------- 共享定义（模板 / 指纹 / 契约） ----------


def card_hash(text: str) -> str:
    """检索面文本指纹：增量同步判"向量是否需要重算"的唯一依据。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_card_text(product: dict) -> str:
    """商品 → 卡片语义文本（v2：不含价格，specs 按键名排序）。

    各段语义职责：
    - 首行 name（category · brand）：BM25 型号/品牌词面精确匹配
    - 配置行：specs 参数型 query 的词面来源（sorted 防字典序漂移假差异）
    - introduction：场景语义主力（dense 路核心）
    价格不进 text：embedding 对数字不敏感，且价格高频变更，
    进 text 会导致每次调价重算 embedding（price 只走标量过滤通道）。
    """
    lines = [
        f"{product['name']}（{product['category']} · {product['brand']}）",
        "配置：" + " / ".join(
            f"{k} {v}" for k, v in sorted(product["specs"].items())
        ),
        *product.get("introduction", []),
    ]
    return "\n".join(lines)


def build_contract(embedder_model: str) -> str:
    """当前构建契约指纹（写入 collection properties.description）。"""
    return f"embedding_model={embedder_model};template={TEMPLATE_VERSION}"


def parse_contract(stored: str) -> dict:
    """解析契约串 → {"embedding_model": ..., "template": ...}；解析失败返回空 dict。"""
    out: dict[str, str] = {}
    for part in (stored or "").split(";"):
        k, _, v = part.partition("=")
        if k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


# ---------- diff（纯逻辑，可单测） ----------


@dataclass
class SyncPlan:
    """一轮同步的差异计划（id 分类 + 每行渲染结果）。"""

    added: list[str] = field(default_factory=list)        # 从未同步 / 状态缺失
    text_changed: list[str] = field(default_factory=list)  # 卡片文本变了 → 重算向量
    row_changed: list[str] = field(default_factory=list)   # 仅行变（如改价）→ 复用向量
    unchanged: list[str] = field(default_factory=list)     # 无变化 → 跳过
    deleted: list[str] = field(default_factory=list)       # Milvus 有、PG 无 → 删除
    texts: dict[str, str] = field(default_factory=dict)    # pid → 渲染后的卡片文本
    new_hashes: dict[str, str] = field(default_factory=dict)  # pid → 新指纹


def plan_sync(snapshot: dict, milvus_ids: set[str]) -> SyncPlan:
    """对完整快照做 diff 五分类（纯函数：不碰 PG / Milvus / embedding）。

    snapshot：pid → 业务字段 + updated_at / card_hash / indexed_at（见
    catalog_repo.load_sync_snapshot）。分类规则：
    - indexed_at / card_hash 缺失       → added（含降级补算场景：BM25 已入库待补向量）
    - card_hash != 新指纹               → text_changed（重算 embedding）
    - updated_at > indexed_at           → row_changed（改价等行变，复用旧向量）
    - 其余                              → unchanged
    - milvus_ids - PG 全集              → deleted（行删了 / 下架了）
    """
    plan = SyncPlan()
    pg_ids: set[str] = set()
    for pid, row in snapshot.items():
        pg_ids.add(pid)
        text = build_card_text(row)
        new_h = card_hash(text)
        plan.texts[pid] = text
        plan.new_hashes[pid] = new_h

        if row.get("indexed_at") is None or row.get("card_hash") is None:
            plan.added.append(pid)
        elif row["card_hash"] != new_h:
            plan.text_changed.append(pid)
        elif row["updated_at"] > row["indexed_at"]:
            plan.row_changed.append(pid)
        else:
            plan.unchanged.append(pid)
    plan.deleted = sorted(milvus_ids - pg_ids)
    return plan


@dataclass
class SyncStats(SyncPlan):
    """同步执行结果：在计划之上追加执行计数。"""

    embedded: int = 0        # 实际调用 embedding 的商品数
    degraded: list[str] = field(default_factory=list)  # embedding 失败降级（仅 BM25 上线）
    upserted: int = 0
    deleted_count: int = 0
    marked: int = 0          # 状态成功写回的行数
    lock_skipped: list[str] = field(default_factory=list)  # 乐观锁未命中（期间又被改）


# ---------- sync 编排（IO 层） ----------


def _get_client():
    """懒加载 MilvusClient（与 search_products 同款：http uri 先 TCP 探活快败）。"""
    from pymilvus import MilvusClient

    ensure_reachable(settings.milvus_uri)
    return MilvusClient(settings.milvus_uri)


def _get_embedder() -> Embedder:
    """懒加载 Embedder（复用知识库同一套 embedding 配置回退链）。"""
    return Embedder(
        api_key=settings.effective_embedding_api_key,
        base_url=settings.effective_embedding_base_url,
        model=settings.effective_embedding_model,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )


def _read_contract(client) -> str:
    """读 collection 契约（properties.description 优先，顶层 description 兜底）。"""
    info = client.describe_collection(settings.product_collection)
    props = info.get("properties") or {}
    return (props.get("description") or info.get("description") or "").strip()


def _collection_dim(client) -> int:
    """从 schema 读 embedding 维度（降级填零向量用）。"""
    info = client.describe_collection(settings.product_collection)
    for f in info.get("fields", []):
        if f.get("name") == "embedding":
            return int(f.get("params", {}).get("dim", 0))
    raise RuntimeError("product_kb schema 中未找到 embedding 字段")


def _check_contract(client, embedder: Embedder) -> None:
    """契约校验：不匹配直接拒绝增量，要求 rebuild。

    比对的是"实际构建时的 embedder 模型"（写入时记录的 embedder.model），
    而非 settings 配置值——回退链下两者可能不同。
    """
    stored = parse_contract(_read_contract(client))
    expected = parse_contract(build_contract(embedder.model))
    for key, want in expected.items():
        if stored.get(key) != want:
            raise RuntimeError(
                f"索引契约不匹配：{key} 索引={stored.get(key)!r} vs 当前={want!r}。"
                f"请执行 python app/scripts/build_product_kb.py --mode rebuild 重建索引。"
            )


def _fetch_milvus_ids(client) -> set[str]:
    """拉取 product_kb 全量主键（删除对账用；千级体量单次 query 足够）。"""
    rows = client.query(
        collection_name=settings.product_collection,
        filter="",
        output_fields=["product_id"],
        limit=16384,
    )
    return {r["product_id"] for r in rows}


def sync(
    *,
    dry_run: bool = False,
    embedder: Embedder | None = None,
    client=None,
) -> SyncStats:
    """执行一轮增量同步。任何失败抛异常，由外壳（CLI / worker）决定如何呈现。

    顺序（develop_docs/rag增量更新.md 5.3）：
    加载校验 → 契约比对 → diff → （dry-run 止步）→ embed+upsert → 删除
    → 抽查对齐 → 最后提交状态（乐观锁）
    """
    from app.db.catalog_repo import load_sync_snapshot, mark_products_indexed

    # 1. 加载完整快照：空快照 = 数据源异常，绝不能当"全删"依据
    snapshot = load_sync_snapshot(settings.database_url)
    if not snapshot:
        raise RuntimeError(
            "PG 商品快照为空（连接失败或未播种）。为防误删索引，本轮中止；"
            "请检查数据库后重试。"
        )

    client = client or _get_client()
    if not client.has_collection(settings.product_collection):
        raise RuntimeError(
            "product_kb collection 不存在，请先执行 "
            "python app/scripts/build_product_kb.py --mode rebuild"
        )

    # 2. 契约比对
    embedder = embedder or _get_embedder()
    _check_contract(client, embedder)

    # 3. diff
    milvus_ids = _fetch_milvus_ids(client)
    plan = plan_sync(snapshot, milvus_ids)
    stats = SyncStats(**plan.__dict__)

    # dry-run：只出计划，不写 Milvus、不写 PG、不调 embedding
    if dry_run:
        return stats

    # 4. 新增 + 文本变：重算向量 → 整行 upsert
    to_embed = plan.added + plan.text_changed
    vectors: dict[str, list[float]] = {}
    if to_embed:
        try:
            encoded = embedder.encode([plan.texts[pid] for pid in to_embed])
            vectors = dict(zip(to_embed, encoded))
            stats.embedded = len(to_embed)
        except Exception:
            # 降级（借鉴 template）：向量填零、BM25 照常上线（服务端 Function
            # 由新 text 生成），这些行不写状态（card_hash 留 NULL）→ 下轮自动补算
            dim = _collection_dim(client)
            zero = [0.0] * dim
            vectors = {pid: zero for pid in to_embed}
            stats.degraded = list(to_embed)

    rows_to_upsert: list[dict] = []
    for pid in to_embed:
        row = snapshot[pid]
        rows_to_upsert.append(_build_row(row, plan.texts[pid], vectors[pid]))

    # 仅行变：text/向量没变 → 点查旧行取向量，price/category 取 PG 新值
    for pid in plan.row_changed:
        old = client.query(
            collection_name=settings.product_collection,
            filter=f'product_id == "{pid}"',
            output_fields=["embedding"],
            limit=1,
        )
        if not old:
            # Milvus 里居然没有（状态说同步过）→ 当新增处理，下轮不补
            stats.added.append(pid)
            continue
        rows_to_upsert.append(
            _build_row(snapshot[pid], plan.texts[pid], old[0]["embedding"])
        )

    # upsert 批量写（pymilvus 单次上限内；百级体量一次写入即可）
    if rows_to_upsert:
        client.upsert(collection_name=settings.product_collection, data=rows_to_upsert)
        stats.upserted = len(rows_to_upsert)

    # 5. 删除失效商品（PG 无此 id；分批防超长请求）
    if plan.deleted:
        client.delete(
            collection_name=settings.product_collection, ids=plan.deleted
        )
        stats.deleted_count = len(plan.deleted)

    # 6. 抽查对齐（2026-09-14 错位事故防线：行数对≠内容对）
    _spot_check(client, snapshot, plan, vectors)

    # 7. 最后提交状态（乐观锁）。降级行不提交（card_hash 留 NULL，下轮补算）
    degraded = set(stats.degraded)
    to_mark = {
        pid: plan.new_hashes[pid]
        for pid in plan.added + plan.text_changed
        if pid not in degraded
    }
    # 仅行变的行：向量没换，card_hash 维持原值，只刷新 indexed_at
    for pid in plan.row_changed:
        to_mark[pid] = snapshot[pid]["card_hash"]

    if to_mark:
        expected = {pid: snapshot[pid]["updated_at"] for pid in to_mark}
        stats.marked, stats.lock_skipped = mark_products_indexed(
            settings.database_url, to_mark, expected_updated_at=expected
        )

    client.load_collection(settings.product_collection)
    return stats


def _build_row(row: dict, text: str, vector: list[float]) -> dict:
    """组装 Milvus 整行：主键 + 检索面 + 标量（一律取 PG 真值）+ 向量。"""
    return {
        "product_id": row["product_id"],
        "text": text,
        "price": row["price"],
        "category": row["category"],
        "embedding": vector,
    }


def _spot_check(client, snapshot: dict, plan: SyncPlan, vectors: dict,
                retries: int = 3, interval: float = 1.0) -> None:
    """抽查 upsert 后的行与预期一致；最终仍不一致才中止（状态未提交，可安全重试）。

    Milvus upsert 后有最终一致可见延迟（segment 落盘/对齐——rebuild 脚本
    sleep(2) 的同款问题），因此带间隔重试而非盲查一次。
    """
    import time

    written = plan.added + plan.text_changed + plan.row_changed
    if not written:
        return
    sample = random.sample(written, min(3, len(written)))
    for pid in sample:
        want_prefix = plan.texts[pid][:20]
        want_price = snapshot[pid]["price"]
        last = ""
        for _ in range(retries):
            got = client.query(
                collection_name=settings.product_collection,
                filter=f'product_id == "{pid}"',
                output_fields=["text", "price"],
                limit=1,
            )
            if not got:
                last = f"{pid} upsert 后查无此行"
            elif not got[0].get("text", "").startswith(want_prefix):
                last = f"{pid} text 错位（got={got[0].get('text', '')[:24]!r}）"
            elif got[0].get("price") != want_price:
                last = f"{pid} price 与 PG 真值不一致（got={got[0].get('price')}）"
            else:
                last = ""  # 本行通过
                break
            time.sleep(interval)
        if last:
            raise RuntimeError(f"抽查失败：{last}")
