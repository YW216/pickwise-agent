-- ============================================================================
-- 创建订单功能：业务表 + 幂等记录表
--
-- 数据库：PostgreSQL（项目真值层，见 app/db/connection.py —— psycopg3）
-- 执行：psql -h localhost -p 15432 -U pickwise -d pickwise -f sql/01_orders_and_idempotency.sql
--
-- 设计要点（先读这段，再看 SQL）：
--
-- 1. 幂等记录表 idempotency_records 不是数据库自带的系统表，是你要自己建的
--    普通业务表。数据库提供的只是"唯一约束 + ON CONFLICT 原子插入"这套机制。
--
-- 2. 幂等键 idempotency_key 的格式约定：
--      {run_id}:{step}:{tool_call_id}
--    - run_id      ：一次用户请求的标识（同一句话重试 = 同一个 run_id）
--    - step        ：ReAct 第几步
--    - tool_call_id：模型这次 function call 的 id（同一轮内唯一）
--    为什么三个都要：模型可能在同一步发多个 create_order 调用，只有 tool_call_id
--    才能区分它们；而跨请求重试时 run_id 不变，所以能命中同一条记录。
--
-- 3. status 三态：processing / succeeded / failed
--    - processing：正在执行（还没落定）——此时重复请求应"稍后重试"而非重跑
--    - succeeded ：已成功——重复请求直接返回缓存结果
--    - failed    ：已失败——允许重试（覆盖为 processing 后重跑）
--
-- 4. expires_at + 定时清理：不加 TTL，这张表会无限膨胀。
--    电商下单场景保留 24 小时足够（同一次请求的重试不会隔天）。
-- ============================================================================


-- ---------------------------------------------------------------------------
-- 业务表：订单
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    order_id    VARCHAR PRIMARY KEY,           -- 订单号，如 ORD-20261001-0001
    user_id     VARCHAR NOT NULL,              -- 下单用户（PickWise 单用户假设下恒为 'default'）
    product_id  VARCHAR NOT NULL,              -- 商品 ID（外键指向 products）
    quantity    INTEGER NOT NULL DEFAULT 1 CHECK (quantity > 0),
    unit_price  INTEGER NOT NULL CHECK (unit_price >= 0),   -- 下单时价格快照（元）
    total_price INTEGER NOT NULL CHECK (total_price >= 0),  -- unit_price * quantity
    status      VARCHAR NOT NULL DEFAULT 'created',         -- created / paid / cancelled
    created_at  TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_orders_user ON orders (user_id, created_at DESC);


-- ---------------------------------------------------------------------------
-- 幂等记录表（去重账本）
--
-- 关键列是 idempotency_key，它是 PRIMARY KEY —— 唯一约束本身
-- 就是"去重"这件事的全部实现：数据库不允许同一个 key 出现两行。
-- 其余列都是"记账"用的：这次调用是谁发起的、成没成、结果是什么。
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS idempotency_records (
    idempotency_key  VARCHAR(128) PRIMARY KEY,   -- 去重的根，见文件头格式约定
    tool_name        VARCHAR(64)  NOT NULL,      -- 哪个工具发起的（'create_order'）
    status           VARCHAR(16)  NOT NULL,      -- processing / succeeded / failed
    result           JSONB,                      -- 成功后缓存的返回值（供重放）
    error            TEXT,                       -- 失败原因（failed 时填）
    created_at       TIMESTAMP    NOT NULL DEFAULT now(),
    updated_at       TIMESTAMP    NOT NULL DEFAULT now(),
    expires_at       TIMESTAMP    NOT NULL       -- 过期后可被清理；重试窗口 = 这个时间
);

-- 清理用索引：WHERE expires_at < now() 走这个索引
CREATE INDEX IF NOT EXISTS idx_idem_expires ON idempotency_records (expires_at);


-- ---------------------------------------------------------------------------
-- 清理函数 + 定时任务（PG 用 pg_cron 扩展；没有扩展就手动/外部定时跑 DELETE）
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION cleanup_idempotency_records() RETURNS void AS $$
BEGIN
    DELETE FROM idempotency_records WHERE expires_at < now();
END;
$$ LANGUAGE plpgsql;

-- 如果装了 pg_cron，取消下面两行注释即可每小时清理一次：
-- CREATE EXTENSION IF NOT EXISTS pg_cron;
-- SELECT cron.schedule('cleanup-idem', '0 * * * *', 'SELECT cleanup_idempotency_records()');
