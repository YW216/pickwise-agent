"""对 standalone 部署执行与 lite 集成测试同一套断言（临时 collection，跑完删除）。

lite 通过不代表 standalone 行为一致（两种部署的 segment 管理与可见性
实现不同）——本脚本把 tests/test_product_kb_incremental.py 的五个用例
原样跑在 standalone 上，验证后才算全量通过（develop_docs/rag增量更新.md 11.2）。

隔离：collection 名 = product_kb + "_verify_tmp"，用完即 drop，不碰真 product_kb；
PG 层替换为内存 state（同 lite 集成测试），不写真库。

用法：python tests/verify_product_kb_standalone.py
前置：deploy/milvus 的 standalone 已启动
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pymilvus import MilvusClient  # noqa: E402

from app.agent.rag.milvus_utils import ensure_reachable, resolve_milvus_uri  # noqa: E402
from app.config.settings import settings  # noqa: E402

VERIFY_COLLECTION = settings.product_collection + "_verify_tmp"

from tests.test_product_kb_incremental import (  # noqa: E402
    build_collection,
    build_env,
    case_bm25_refreshed_on_text_change,
    case_delete_not_retrievable,
    case_degrade_then_recover,
    case_scalar_filter_uses_new_value,
    case_upsert_same_id_no_duplicate,
)


def main() -> None:
    uri = resolve_milvus_uri()
    ensure_reachable(uri)
    client = MilvusClient(uri)

    if client.has_collection(VERIFY_COLLECTION):
        client.drop_collection(VERIFY_COLLECTION)

    original_collection = settings.product_collection
    settings.product_collection = VERIFY_COLLECTION  # 用例与 sync 全部走临时 collection
    try:
        build_collection(client)
        cases = [
            case_upsert_same_id_no_duplicate,
            case_bm25_refreshed_on_text_change,
            case_scalar_filter_uses_new_value,
            case_delete_not_retrievable,
            case_degrade_then_recover,
        ]
        for case in cases:
            env = build_env(client, {})
            case(env)
            env.restore()
            print(f"  ✅ {case.__name__}")
        client.drop_collection(VERIFY_COLLECTION)
        print("🎉 standalone 全部用例通过（临时 collection 已清理）")
    except Exception:
        client.drop_collection(VERIFY_COLLECTION)
        raise
    finally:
        settings.product_collection = original_collection


if __name__ == "__main__":
    main()
