import os

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """项目配置，从 .env 文件读取"""

    # 无默认值：密钥类字段一律留空，由 .env / 环境变量注入。
    # 历史教训——这里曾硬编码过一个真实 API Key 并随 commit 进入 git 历史，
    # 公开仓库一旦 push 即可被挖出。故所有凭据字段禁止写默认值。
    # 空串的实际影响由使用点承担：需要凭据的地方（见 assert_*_configured）显式报错，
    # 不需要凭据的地方（纯离线单测、工具 schema 校验）照常工作——
    # 若改成 pydantic 必填校验，13 个 pytest 文件会在无 .env 环境下集体导入失败。
    openai_api_key: str = ""
    openai_base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-v4-flash"
    temperature: float = 0.7
    # 思考强度（DeepSeek V4 思考模式参数，透传 API 的 reasoning_effort）：
    # low=轻思考 / high=日常开发推荐 / max=官方对 Agent 场景的建议档。
    # 仅作用于子 Agent 与 Result 的生成调用；Router 不受此配置影响——它是轻量
    # 分类任务，在 router.py 中固定 low + max_tokens=2048（取值理由见该处注释）。
    # 空串 = 不传该参数（用服务端默认行为）。
    reasoning_effort: str = "high"

    # LLM HTTP 阶段超时，不是整次 chat/Agent 的 deadline。
    # SDK 2.28.0 默认 read=600s；显式设 60s，避免一次等待过长。
    # max_retries=2 表示首次请求 + 最多两次重试，叠加退避后可能超过 60s。
    # SDK 负责连接/超时、408/409/429、>=500 等传输故障；应用不重复叠加。
    openai_timeout: float = Field(default=60.0, gt=0)
    # LLM HTTP 重试不会执行本地写工具，也不会生成应用层 run_id。
    # 写工具需要业务幂等/执行记录；Agent 重放风险与此配置分开治理，
    # 不能把“关闭 LLM 重试”当作避免重复下单/退款的充分条件。
    openai_max_retries: int = Field(default=2, ge=0)

    # ReAct 循环
    max_react_steps: int = 10

    # MCP 配置
    mcp_enabled: bool = False
    mcp_server_url: str = "http://127.0.0.1:9123/mcp"

    # 可观测性（Langfuse 追踪；2026-09-15 .env 已配置，字段先声明占位——
    # pydantic-settings 默认 extra_forbidden，未声明的 .env 键会让整个 Settings 加载失败）
    langfuse_base_url: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # RAG 配置（第5期）
    # Embedding 服务可独立配置（如硅基流动 SiliconFlow）。
    # 三项各自留空时回退到主模型配置——由 effective_embedding_* 三个属性统一收口，
    # 调用方只读那三个，不要直接读这里的原始字段。
    embedding_model: str = "BAAI/bge-m3"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    kb_dir: str = "app/agent/rag/knowledge"
    # Milvus 向量后端：本地开发 milvus-lite（uri=本地 .db 路径），
    # 生产 standalone（uri=http://host:19530），同一套代码仅切换 uri
    milvus_uri: str = "app/sessions/milvus_lite_kb.db"
    milvus_collection: str = "ecom_kb"
    # 商品语义检索库（search_products）：与知识库分 collection——语义域不同
    # （知识问答 vs 商品卡片），更新节奏也不同（指南低频 / 商品信息随目录变）
    product_collection: str = "product_kb"

    # Memory 配置（第7期）
    memory_enabled: bool = True
    memory_dir: str = "app/sessions/memory"
    memory_user_id: str = "default"
    max_ltm_facts: int = 50

    # 真值层数据库（PostgreSQL）：商品目录 / 评价 / 保修的唯一事实源，
    # Milvus 检索索引由它构建。空串 = 未落库，回退本地文件（mock_data + extra JSON）
    database_url: str = "postgresql://pickwise:pickwise@localhost:15432/pickwise"

    # Skill 配置（第8期；2026-09-03 PickWise 化：product-recommend 归 guide，
    # catalog 按 Agent 归属过滤——客服 process-return/track-order 留库不暴露）
    skills_enabled: bool = True
    skills_dir: str = "app/agent/skills/definitions"

    # Evaluation 配置（第9期，离线评估工具，无聊天开关）
    eval_dataset_path: str = "app/evaluation/cases.json"
    # LLM judge（P6 第二步）：eval_use_judge 是全局急停开关（用例还需在 cases.json
    # 里声明 judge_aspects 才会真正触发）；judge 分 advisory，不参与 pass/fail，
    # eval_pass_threshold 留给"judge 升格为门槛"的第三步使用。
    eval_use_judge: bool = True
    eval_pass_threshold: float = 0.6

    # 多轮对话管理
    session_path: str = "app/sessions/session.json"
    # context_window 是策略阈值（deepseek-v4-flash 真实窗口 1M）：提前压缩控制成本与延迟，
    # 非硬件上限。当前按"选购助手会话"形态取 200K——单会话 10-30 轮，压缩极少触发；
    # 若演变为跨天陪伴式会话，上调至 500K 并同步放大 keep_recent_tokens（2026-09-07 裁定）。
    # reserve_tokens 同时是子 Agent/Result 生成调用的 max_tokens——
    # 推理模型的思考与正文共享该预算，默认给足 32K 防思考被截断（P1 5.1 教训的放大版）。
    context_window: int = Field(default=200000, gt=0)
    reserve_tokens: int = Field(default=32768, gt=0)
    compaction_enabled: bool = True
    keep_recent_tokens: int = Field(default=8000, gt=0)
    # summary_max_tokens 身兼两职：摘要生成的 max_tokens（防 LLM 输出被截断），
    # 以及摘要长度的 token 校验上限。2026-09-29 真调实测撞墙：第一次压缩生成的
    # 摘要 1701 字（约 1560 token）已逼近 2048，第二次压缩必然超限失败，
    # 此后摘要只增不减 → 永久压不动 → 上下文一路涨到溢出。提到 4096 留出空间。
    summary_max_tokens: int = Field(default=4096, gt=0)
    # 摘要长度的验收标准（字符数，人可读）。原值 12000 实际是死代码——token 校验
    # （2048）恒先触发，它永远轮不到。降到 3000 让它成为真正的长度约束，
    # 与 prompt 里"整体控制在 1500 字以内"的引导配合（留一倍余量）。
    summary_max_chars: int = Field(default=3000, gt=0)
    tool_result_max_chars: int = Field(default=12000, ge=256)
    # 单条用户输入的估算上限（入口体量闸门）。
    # 必要性：当前 user 消息永远不在压缩范围内（_try_compact 的 active_user_index
    # 把它排除在外），所以"输入本身超长"压缩救不了，必须在进入历史前拦下。
    # 取值依据：触发线 167232、实测轮初 ≤30K、单轮增长 1~3K——16K 不影响正常提问
    # （实测几十到几百 token），且留足 130K+ 余量。
    # 注：入口的完整校验（prompt 注入等）另立专项，此处只做体量闸门。
    max_user_input_tokens: int = Field(default=16000, gt=0)

    @model_validator(mode="after")
    def validate_context_budget(self):
        if self.reasoning_effort and self.reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("reasoning_effort 仅支持 low / high / max（空串 = 不传参数）")
        if self.reserve_tokens >= self.context_window:
            raise ValueError("reserve_tokens 必须小于 context_window")
        if self.keep_recent_tokens + self.summary_max_tokens >= self.context_window - self.reserve_tokens:
            raise ValueError("保留历史和摘要预算必须小于可用上下文窗口")
        return self

    # protected_namespaces 置空：允许 model_name 等 model_ 前缀字段（消除 pydantic v2 警告）
    model_config = {"env_file": ".env", "protected_namespaces": ()}

    # ---------- 凭据完整性检查（2026-10-09）----------
    # 为什么放在这里而不是用 pydantic 必填校验：Settings 在 import 期实例化，
    # 必填字段会让无 .env 的场景（CI 跑离线单测、只读 schema 的工具校验）
    # 直接 ImportError。改成显式断言——用不到的路径不受影响，用得到的路径给出可操作的报错。
    _MISSING_CREDENTIAL = "缺失，请在 .env 中配置 {key}（参考 .env.example）"

    # ---------- Embedding 凭据回退（2026-10-09）----------
    # 为什么需要：早期注释声称「留空则复用 openai_base_url/openai_api_key」，
    # 但 Embedder 构造处是`api_key=settings.embedding_api_key` 直接传值，
    # 空串会被原样送进 OpenAI 客户端 → 401。注释描述的回退从未被实现。
    # 这三个属性是回退的唯一收口点，调用方一律读它们。
    #
    # 为什么 model 也参与回退：provider 与 embedding 模型是绑定的。
    # DeepSeek 官方 API 不提供 embedding 服务（截至 2026-08 定价页仅有对话模型），
    # 所以「主模型配了DeepSeek」时回退出的 base_url 指向一个没有 embedding
    # 端点的host —— 此时若仍沿用默认的 BAAI/bge-m3，请求会以404/模型不存在告终。
    # 配了 EMBEDDING_BASE_URL 就意味着换了 provider，模型名必须跟着换。
    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.openai_api_key

    @property
    def effective_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.openai_base_url

    @property
    def effective_embedding_model(self) -> str:
        #只有换了 provider 才需要换模型名；同一 provider（留空回退）时沿用 embedding_model。
        # 留空兜底为 OpenAI 官方模型名——回退只在主模型 provider 确实提供
        # embedding 端点时成立（OpenAI 或兼容网关等）。
        if self.embedding_base_url:
            return self.embedding_model
        return self.embedding_model or "text-embedding-3-small"

    def assert_openai_configured(self) -> None:
        """需要调用主 LLM 时使用：确认主模型凭据已配置。"""
        if not self.openai_api_key:
            raise RuntimeError(
                self._MISSING_CREDENTIAL.format(key="OPENAI_API_KEY")
            )

    # 已知无 embedding 端点的 provider 主机名片段（实测 DeepSeek 返回 404，
    # 官方定价页截至 2026-08 也只列对话模型）。用于把「回退不可用」
    # 从运行时的 404 提前到配置期的显式提示。
    _NO_EMBEDDING_HOSTS = ("deepseek.com",)

    def assert_embedding_configured(self) -> None:
        """需要向量化时使用。

        分三档：独立配置可用 / 回退可用 / 回退不可用。
        最后一档是实测踩出来的——DeepSeek 官方不提供 embedding 服务，
        只配主模型时凭据「看起来齐了」，实际请求 404。这种失败必须在配置期
        说清楚，否则用户会以为是 key 填错了。
        """
        if self.embedding_base_url:
            if not self.embedding_api_key:
                raise RuntimeError(
                    "已配置 EMBEDDING_BASE_URL 但缺少 EMBEDDING_API_KEY，"
                    "无法完成向量化（指定了独立 provider 就必须给它配密钥）"
                )
            return

        # 回退路径：依赖主模型凭据
        if not self.openai_api_key:
            raise RuntimeError(
                "Embedding 凭据缺失：请配置 EMBEDDING_API_KEY / EMBEDDING_BASE_URL，"
                "或配置 OPENAI_API_KEY 走回退"
            )
        if any(h in self.openai_base_url for h in self._NO_EMBEDDING_HOSTS):
            raise RuntimeError(
                f"主模型 provider（{self.openai_base_url}）不提供 embedding 服务，"
                "回退路径不可用（实测返回 404）。请显式配置 EMBEDDING_BASE_URL "
                "与 EMBEDDING_API_KEY 指向支持 embedding 的服务，"
                "例如硅基流动 https://api.siliconflow.cn/v1 + BAAI/bge-m3"
            )


settings = Settings()

# ---------- Langfuse 凭证桥接（2026-09-16） ----------
# 为什么需要：langfuse 的 OpenAI 补丁是 wrapt **类级全局补丁**——只要进程 import 了
# app.multi_agent（→ orchestrator → from langfuse.openai import openai），所有 OpenAI
# 调用都会被接入；而 SDK 只从 os.environ 读凭证，pydantic 的 env_file 只把 .env 读进
# settings 对象。未桥接的后果：未被 load_dotenv 的入口（router/rag 评测、build_* 脚本）
# 里 SDK 取不到 public_key → 静默禁用上报 + 每次调用打印一条告警。
# 这里统一下沉到 settings：任何入口 import settings 即生效（setdefault 保证 shell 显式变量优先）。
# 注意 SDK 同时认 LANGFUSE_BASE_URL 与 LANGFUSE_HOST（langfuse/_client/client.py），
# 项目 .env 用的是带 base_url 字段的 LANGFUSE_BASE_URL。
for _key, _value in (
    ("LANGFUSE_BASE_URL", settings.langfuse_base_url),
    ("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key),
    ("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key),
):
    if _value:
        os.environ.setdefault(_key, _value)
