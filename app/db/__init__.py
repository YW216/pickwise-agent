"""PostgreSQL 真值层数据访问包（2026-09-15）。

真值层（PG）与检索层（Milvus）分离后，数据库访问统一收口于此：
- connection.py：连接与连通性探测
- catalog_repo.py：商品目录仓储（结构化过滤查询 / 按 ID 批查 / 全量加载 / 建表播种）

分工约定：
- search_catalog（工具）→ catalog_repo.search_candidates：SQL 直查，确定性过滤，
  PG 数据变更即时可见
- search_products（工具）→ Milvus 索引检索（索引由真值构建），命中后按
  product_id 组装 brief（真值快照）
- 评价入库暂缓：REVIEWS 暂留本地 mock_data，落库解冻后在此包新增 reviews_repo
"""
