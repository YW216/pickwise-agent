"""用户收藏迁移：python -m app.scripts.migrate_user_state（可重复运行）。"""

import argparse

from app.config.settings import settings
from app.db.user_repo import init_user_schema


def main():
    parser = argparse.ArgumentParser(description="建用户收藏表，并一次性导入原 CLI 用户收藏")
    parser.add_argument("--legacy-user-id", default=settings.memory_user_id)
    args = parser.parse_args()
    imported = init_user_schema(settings.database_url, legacy_user_id=args.legacy_user_id)
    print(f"用户收藏迁移完成：导入 {imported} 条，归属用户 {args.legacy_user_id}")


if __name__ == "__main__":
    main()
