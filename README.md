# PickWise · 选购助手「小P」

> 一个用纯 Python 编排的 **多Agent 3C 选购决策助手**。不依赖 LangChain / LangGraph，手写 ReAct 循环与多 Agent 编排，配双层自动化评估体系。

CLI 交互入口，纯后端项目（无 Web 前端）。

---

## 核心设计：一条路径的编排

单 Agent / 多 Agent **不是两套模式**，而是同一条路径的两个特例 —— N=1 时直返，N>1 时走汇总。

```
用户输入
   │
   ├─ 1. 构建 ContextPack（历史 / 摘要 / 记忆 / 技能目录 / 商品记忆）
   ├─ 2. 上下文压缩（事务式，失败零副作用）
   ├─ 3. Router 判定场景 → ["presale"] / ["consult"] / 两者
   ├─ 4. 场景内并行执行（ThreadPoolExecutor）
   │       presale  售前Agent · 8 个工具
   │       consult  咨询 Agent · 3 个工具
   ├─ 5. N=1 → 直接返回N>1 → Result Agent 汇总黑板
   └─ 6. 记忆更新 → 落盘

返回纯文本 reply
```

### 场景与Agent

| Agent | 场景 | 工具数 | 工具清单 |
|---|---|---|---|
| `presale` | 售前选购 | 8 | `search_catalog`、`get_user_favorites`、`search_products`、`get_detail`、`compare_products`、`retrieve_knowledge`、`load_skill`、`recall_user_memory` |
| `consult` | 售前咨询 | 3 | `retrieve_knowledge`、`get_detail`、`load_skill` |

**工具白名单是硬隔离**（按 Agent 维度裁剪 function schema），不是靠 prompt 请求模型自觉。Router 规则写在 prompt 里，Python 侧只做输出解析与固定顺序重排，保证结果可复现。

### 三个不那么显然的设计

**1. 历史视图投影** —— 共享历史里含其他 Agent 的 `tool_calls`，模型看到邻居用过某工具会模仿调用，而执行层必然拒收，白费两轮 LLM。解法是按「整块」（一条 `assistant(tool_calls)` + 其后连续 tool 消息）判定，块内工具全部越权则折叠为中性文本。
折叠文案刻意不带工具名，且按语义类别（检索 / 对比 / 用户偏好）生成 —— 把「读取用户收藏夹」说成「完成了一次信息检索」，会让下一轮 Agent 形成错误认知而跳过应有的检索。**文案失真诱发的幻觉，比泄漏工具名更贵。**

**2. 上下文压缩是事务** —— 先压缩（纯计算）→ 确认确实缩小了 → 才删除前缀并替换 summary。中途任何一步失败都保持原状态。切点从最新消息往回累积 token，只在 user turn 起点切，天然不拆开 `user → assistant(tool_calls) → tool → assistant` 链。摘要撞墙时自压缩摘要，宁可丢叙述性细节也不让压缩整体失效。

**3. HTTP 成功 ≠ 任务成功** —— 拿到 HTTP 200 不算数：内容为空、`finish_reason=length`（被截断）、还挂着 `tool_calls`（模型还想调工具），三者任一都判定为「模型未返回完整有效的最终答复」并降级，不允许把不完整输出当答案交付。

---

## 快速开始

**唯一必需**：一个 OpenAI 兼容协议的 API Key。其余服务都有降级路径。

### 1. 安装

```bash
git clone https://github.com/YW216/PickWise.git
cd PickWise

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install langfuse pytest                        # 见下方「关于依赖」
```

### 2. 配置

```bash
cp .env.example .env
```

最小配置（其余全部有默认值）：

```ini
OPENAI_API_KEY=sk-你的key
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-v4-flash

EMBEDDING_API_KEY=sk-你的embeddingkey
EMBEDDING_MODEL=BAAI/bge-m3
```

> ⚠️ **不要照抄 `.env.example` 里所有键**。本项目 `Settings` 用 pydantic-settings的 `extra_forbidden`，`.env` 里出现**未声明的键会让整个配置加载失败**。`.env.example` 中 `RAG_BACKEND`、`KB_INDEX_PATH`、`CHROMA_PERSIST_DIR`、`CHROMA_COLLECTION`、`MULTI_AGENT_ENABLED` 属于历史遗留，删掉即可。

### 3. 运行

```bash
python main.py
```

交互式会话，支持 4 个内置命令：

| 命令 | 作用 |
|---|---|
| `skills` | 列出已加载技能 |
| `memory` | 查看短期 / 长期记忆原文 |
| `reset` | 重置对话 |
| `quit` / `exit` | 退出（自动落盘会话与记忆） |

---

## 外部服务：全部可选

| 服务 | 用途 | 必需 | 不配会怎样 |
|---|---|---|---|
| **LLM API** | 推理 | ✅ **必需** | 无法启动 |
| **Milvus** | 向量检索 | 功能必需 | 默认走 **milvus-lite**（本地 `.db` 文件），**零部署** |
| **PostgreSQL** | 商品真值层 | 可选 | 回退本地 JSON 数据文件 |
| **Langfuse** | 调用链追踪 | 可选 | 全链路 no-op |

**开发环境开箱即用**，因为默认就是 milvus-lite。需要 standalone 时改一个配置项即可，同一套代码：

```ini
MILVUS_URI=http://localhost:19530      # 本地文件 → standalone 服务
```

Milvus 服务端用官方的独立 compose（`milvus standalone + etcd + minio`），PostgreSQL 在另一个 compose 里，两者**刻意不合并**——真值层与检索层分离，真值变更后重建索引即可。

---

## 工具体系

8 个工具，**OpenAI 原生 function calling**，schema 单源：

```
TOOL_DEFINITIONS ──派生──→ _SCHEMA_MAP ──→ 执行前参数闸门
     │                        │
     └──── 给模型看的说明书 ←──┘ 同一份
```

校验器反读 schema 约束做拦截，模型看到的与执行层校验的永远同源，不会漂移。参数不合契约时**回喂给模型自己改**，而不是直接报错终止。

三道闸门按序收口：协议层 JSON 解析 → 契约层 schema 校验 → **重复调用熔断**（同一签名的工具第 3 次调用被拦下，返回「换查询条件或基于现有信息作答」这类可操作提示，而不是硬失败）。

熔断签名按参数归一化后的稳定 JSON 计算，所以模型换个写法（`"10"` vs `10`）绕不过去。

---

## 检索：dense + BM25 混合

知识库与商品库都走 Milvus 混合检索，**服务端 RRF 按排名融合**：

```
query 原文 ──────────────→ BM25 稀疏路（服务端 jieba 分词建索引）
         └→ Embedding ──→ dense 路（COSINE）
                              ↓
                        RRFRanker融合
```

原文一路直通 BM25 这一步是刻意的：只有 retriever 层同时持有原文和 embedder，所以透传放在这一层而非上层。混合检索的生产路径与评测路径共用同一实现，保证评测结果不分叉。

知识库与商品库分两个collection：语义域不同（问答vs 商品卡片），更新节奏也不同。

---

## 评估体系

**双层设计：规则断言判生死，LLM-as-judge 只参考。**

`CaseResult.passed` **只看规则**，judge 打分完全不影响通过判定。这样做是为了让门禁稳定可复现 —— 评审模型的输出天然有抖动，不能拿它当开关。

### 跑起来

```bash
python app/scripts/run_eval.py# 端到端，20 条
python app/scripts/run_eval.py --case-id <id>     # 单条调试
python app/scripts/run_eval.py --self-test       # 判分器自检
python app/scripts/run_router_eval.py            # Router 专项，55 条，可设门禁退出码
python app/scripts/run_rag_eval.py --top-k 5     # 检索 Hit@K / MRR / Recall@K
```

### 用例集（共 231 条）

| 数据集 | 条数 | 用途 |
|---|---|---|
| `cases.json` | 20 | 端到端（含多轮指代消解） |
| `router_cases.json` | 55 | Router 场景判定，带 `group` / `difficulty` 维度 |
| `rag_cases.json` | 152 | 检索质量，只取人工审核过的标注 |
| `cases_isolation.json` | 4 | 工具白名单泄漏探针 |

端到端用例支持多轮（`turns`），`expected_route` 支持 `"presale|consult"` 表示两者皆可。

### 两个值得单说的地方

**Sandbox 隔离**：每条用例独立临时 session、**关闭记忆读写**（否则历史 LTM 会注入 prompt 污染评分）、关闭 MCP只用本地 mock 保证 ground truth 确定。插桩只给共享的 `chat.completions.create` 打 monkey-patch，不改生产代码一行。

**判分器自己也被测**：`--self-test` 会构造「回复含伪造ID / 伪造价格」的轨迹，断言判分器必须能抓出来。判分器不可信则整个评估体系无意义。

### Judge 的防抖动设计

温度固定 0 + rubric 版本号锁定 + 强制 JSON 输出，解析失败重试一次，再失败返回「未判」而非打假分。未判样本原始输出落盘，并按 `finish_reason` 区分`length`（该加预算）与 `stop`（该改 prompt）—— 两类问题的修法完全不同。

---

## 韧性设计

### 职责边界

| 层级 | 负责 | **不**负责 |
|---|---|---|
| SDK HTTP | 超时 / 重试 / 退避 | 执行本地写工具 |
| Router | 瞬时故障退回默认场景 | 保证降级后仍完整完成推荐 |
| SubAgent | 只接受非空且未截断的答复 | 把代码异常伪装成正常答案 |
| Result | 整合已有成果 | **编造**未成功 Agent 的答案 |
| 入口 | 接纳 / 失败轮补答 | 因记忆或保存失败覆盖已生成的答案 |

**部分失败不整体失败**：单 Agent 失败直接抛错；多 Agent 失败则记黑板、由 Result 如实告知。故障 Agent 不重复调用，只有 `overflow` 子集在压缩成功后重试一次（压不动就不重跑，上下文一样大必然二次溢出）。

### 异常分类按两个正交维度

`is_transient`（连接 / 超时 / 限流 / 5xx）与 `is_context_overflow`（`context_length_exceeded`）刻意分开，且都收窄识别范围 —— **宁可漏判也不误判**。混为一谈会出事：400 参数错误压缩无用，连接中断压缩也无用。

### 一条安全边界

LLM HTTP 重试**不会执行本地写工具**。写工具需要业务幂等与执行记录，**不能把「关闭 LLM 重试」当作避免重复下单 / 退款的充分条件**。

---

## 项目结构

```
app/
├── multi_agent/        编排层
│   ├── orchestrator.py     唯一编排器
│   ├── router.py           LLM 场景判定 + 固定顺序重排
│   ├── agents.py           轻量 ReAct SubAgent
│   ├── context_pack.py     共享上下文 + 历史视图投影
│   └── blackboard.py       单轮回执收集
├── agent/     能力层（无编排）
│   ├── compaction.py       上下文压缩（纯函数）
│   ├── context_budget.py   token 预算 / 异常分类
│   ├── product_tracker.py  商品记忆
│   ├── tools/              8 个工具 + schema + 校验闸门
│   ├── memory/             STM / LTM + LLM 事实提取
│   ├── rag/                Milvus 混合检索
│   └── skills/             SKILL.md 流程加载
├── evaluation/  评估（sandbox / metrics / judges / reporter）
├── config/settings.py      pydantic-settings
└── scripts/                三个评测入口

main.py                 CLI 入口
tests/                13 个 pytest + 10 个 verify 脚本
deploy/                  Milvus / PostgreSQL compose（gitignore）
```

`app/multi_agent/` 是编排，`app/agent/` 是能力，两层职责不重叠。

### 记忆

| | 存储 | 更新时机 |
|---|---|---|
| **STM** 短期 | 内存 + session 文件恢复 | 每轮成功后，只看最近 6 条消息 |
| **LTM** 长期 | `{memory_dir}/{user_id}.json`，原子写入 | **会话关闭时**才巩固 |

LTM 注入时附带最近 3 条交互摘要。STM 提取失败不影响已生成的答案，LTM 提取失败不阻止资源清理 —— **附属工作永远不能覆盖主答复**。

---

## 测试

```bash
pytest tests/ -v
```

**刻意完全离线**，不烧 token、不访问外网：`test_resilience.py` 用假 client 注入故障，`test_llm_transport.py` 用 `httpx.MockTransport`，`test_evaluation.py` 不调 LLM。故障降级路径要能测，就不能依赖真调用。

另有 10 个 `verify_*.py` 脚本是真调 LLM / Milvus 的端到端手工验证，需手动执行。

---

## 技术栈

| 层 | 选型 |
|---|---|
| 编排 | **自研**（手写 ReAct + ThreadPoolExecutor，零 Agent 框架） |
| LLM | `openai` SDK（OpenAI 兼容协议，实接 DeepSeek v4-flash） |
| Embedding | `BAAI/bge-m3`（可切任意 OpenAI 兼容服务） |
| 向量库 | Milvus（milvus-lite / standalone 同构切换）+ BM25 + RRF |
| 数据 | PostgreSQL 16（可选） |
| 追踪 | Langfuse（可选，wrapt 类级补丁） |
| 校验 | pydantic v2 + pydantic-settings |
| 测试 | pytest + `unittest.mock` + `httpx.MockTransport` |

**7 个直接依赖**，没有 LangChain / LangGraph / LlamaIndex。

---

## 已知事项

- **`.env.example` 含失效键** —— `RAG_BACKEND`、`CHROMA_*`、`KB_INDEX_PATH`、`MULTI_AGENT_ENABLED` 在代码中已无对应字段（向量库已统一切到 Milvus，单/多Agent 双模式已合并）。复制后请删除这些键，否则 `extra_forbidden` 会让配置加载失败。
- **`requirements.txt` 未声明 `langfuse` 与 `pytest`** —— `langfuse` 在 `orchestrator.py` 是强 import，不装无法启动；`pytest` 是跑测试的前提。两者需单独安装。
- **`embedding_model` 三处不一致** —— `settings.py` 默认 `BAAI/bge-m3`，`embedder.py` 类默认与 `.env.example` 写 `text-embedding-3-small`。实际生效取决于 `.env`。
- **MCP 服务端已暂停** —— `mcp_server/server.py` 只剩注释，客户端代码就绪但服务端工具已下线，当前降级使用本地工具。
- **根目录 `agentdemo.py` / `test.py` / `wtest.py`** 是学习期function calling 与装饰器的草稿，不属于应用，`agentdemo.py` 本身无法运行。
- **`tests/test_multi_agent.py`** 是脚本式断言（非 pytest 风格），不被 pytest 收集。