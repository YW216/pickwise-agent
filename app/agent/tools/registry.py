"""工具注册表：OpenAI function calling schema + 分发执行（PickWise）。

结构仿 ecom：
- TOOL_DEFINITIONS：给模型看的工具说明书（手写 OpenAI function schema，兼作校验规则源）
- _TOOL_MAP       ：工具名 → 业务函数
- _SCHEMA_MAP     ：工具名 → parameters（校验闸门的规则源，从上一项派生）
- execute_tool    ：统一分发 + 契约校验 + 异常兜底

错误处理（前两处是"闸门"——拒绝执行并把原因回喂模型，第三处是兜底）：
1. 工具函数内：业务失败的人话文案（如"商品不存在"）
2. validation 契约校验：参数不合签名 → 不执行，返回可操作原因（模型改值重调）；
   被归一化或夹紧时给信封追加 notice——削过的结果必须让模型知道
3. 异常兜底：漏网异常 → print 排查 + 拼异常的 error

schema 编写约定：

1. 体积：每轮请求都会重发 schema，体积直接影响 token 成本。
   description 只写「做什么 + 返回什么 + 参数怎么填」，场景示例一律不放。

2. 边界（重要）：可以写「何时用我」和「我不是什么」这类**自我界定**，
   但**禁止写「去用别的工具」「别用 A 代替 B」这类跨工具路由**。
   原因：同一工具会被多个 Agent 共享而白名单不同，指向别的工具的指令
   ——对没有那个工具的 Agent 是噪音，更糟时会诱导它调用看不见的工具名。
   路由规则统一写在 app/prompts/agents.py 各 Agent 的「工具使用原则」里。

3. 取值固定的参数用 enum 表达，既省字数又带一层值域约束。

4. 约束关键字（type / enum / required / minimum / maximum / minItems /
   maxItems）是**双用**的：模型看它一次给对，执行层（validation 模块）反读它
   做闸门——规则只有这一份，不另建校验表，避免"说明书改了、闸门忘改"。
   例外：**不写 additionalProperties**。校验器默认过滤未知参数并回传提示，
   比每条 schema 挂着 additionalProperties: false 更省每轮重发的 token。
"""

import json
from typing import Callable

from app.agent.tools.catalog import search_catalog
from app.agent.tools.detail import compare_products, get_detail
from app.agent.tools.favorites import get_user_favorites
from app.agent.tools.knowledge import retrieve_knowledge
from app.agent.tools.memory_tool import recall_user_memory
from app.agent.tools.result import fail
from app.agent.tools.search_products import search_products
from app.agent.tools.skill_tool import load_skill
from app.agent.tools.validation import ArgumentError, validate_arguments

# 工具名 → 业务函数。
# 所有工具函数签名均为 (参数...) -> dict（返回统一信封），
# 因此用 Callable[..., dict] 比裸 Callable 更精确（Callable 等价于 Callable[..., Any]）。
# 机制型工具（load_skill / recall_user_memory）由 Agent 初始化时通过
# set_skill_manager / set_memory_manager 注入依赖，注册表只做分发。
_TOOL_MAP: dict[str, Callable[..., dict]] = {
    "search_catalog": search_catalog,
    "get_user_favorites": get_user_favorites,
    "search_products": search_products,
    "get_detail": get_detail,
    "compare_products": compare_products,
    "retrieve_knowledge": retrieve_knowledge,
    "load_skill": load_skill,
    "recall_user_memory": recall_user_memory,
}

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": (
                "结构化过滤检索（精确仪器）：参数与 PG 表列一一对应——query 匹配"
                "商品名字面（ILIKE）、brand 品牌精确匹配、category 品类精确、"
                "budget_max 价格数值上限；不匹配 specs 参数值与介绍文案（参数"
                "条件如「16GB」「独显」、语义/模糊需求由 search_products 承担）。"
                "适用于：品牌/型号字面查询、品类+价格边界筛选、穷举在售清单。"
                "返回全部通过过滤的候选。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "型号字面关键词，仅匹配商品名，如「凌霄 Pro 14」",
                    },
                    "brand": {
                        "type": "string",
                        "description": (
                            "品牌精确匹配（品牌罗列/品牌筛选时使用）。品牌是开放"
                            "数据（当前已知：星海/曜石/云章/墨白/极光，以目录"
                            "实际返回为准），不设 enum——新增品牌不应改代码"
                        ),
                    },
                    "category": {
                        "type": "string",
                        "description": "品类",
                        "enum": ["笔记本", "手机", "耳机"],
                    },
                    "budget_max": {
                        "type": "number",
                        "description": "预算上限（元）",
                        "minimum": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "返回条数，默认 10。上限 20——用户索要的清单超过上限时"
                            "按上限返回即可，返回体中的 matched 会给出符合条件总数，"
                            "据此提醒用户结果被截断并请其补充条件，不要逐款查详情"
                        ),
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_favorites",
            "description": "查询用户收藏夹商品，用于「我的收藏」「收藏的那款」。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "品类过滤",
                        "enum": ["笔记本", "手机", "耳机"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回条数，默认 20",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "语义搜索商品（默认主搜索）：卡片向量覆盖商品全部字段"
                "（名称/品牌/参数/介绍），自然语言需求、模糊描述、口碑场景、"
                "型号、品牌皆宜；支持预算/品类标量过滤。返回最相似的 limit 条"
                "轻卡片（按相似度排序，非穷举）。需要穷举在售清单或硬条件"
                "全部满足的精确筛选时配合 search_catalog。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "自然语言描述的购物需求，如「适合打游戏的轻薄本」"
                            "「通勤戴的安静耳机」，或直接给型号「凌霄 Pro 14」"
                        ),
                        "minLength": 1,
                    },
                    "category": {
                        "type": "string",
                        "description": "品类过滤",
                        "enum": ["笔记本", "手机", "耳机"],
                    },
                    "max_price": {
                        "type": "integer",
                        "description": "预算上限（元），如 5000 表示只要 5000 元及以下的商品",
                        "minimum": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "返回商品数上限，默认 5，上限 20。语义检索按相似度取前 N 条，"
                            "返回条数触达上限意味着可能还有更多相关商品，应提醒用户"
                            "补充条件缩小范围"
                        ),
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_detail",
            "description": (
                "按 product_id 查单个商品的价格与完整参数。适用于用户指名少数"
                "几款、需要比对具体参数的场景（可一次批次内并发多款）。"
                "批量罗列某品类清单时不必逐款调用：search_catalog 的候选卡片"
                "已含名称/品牌/价格/定位，足以展示可选范围。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "商品 ID，如 LP-01",
                        "minLength": 1,
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": "对比 2-4 款商品的同维度参数，一次传入全部商品 ID。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "description": "商品 ID 列表，2-4 个，如 ['LP-01','LP-02']",
                        "minItems": 2,
                        "maxItems": 4,
                    },
                },
                "required": ["product_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": (
                "检索平台知识库：选购指南（参数科普/场景适配）、售后政策（保修/退换货）、配送说明、会员权益、常见问题 FAQ。"
                "基于检索结果作答，勿编造。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用用户原问题或一句中文描述知识点",
                        "minLength": 1,
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回片段数，默认 3",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "加载指定技能的完整流程指令。当用户问题匹配「可用技能」目录中"
                "某个技能的场景时调用，加载后按指令流程处理。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "技能名，见系统提示中的可用技能目录",
                        "minLength": 1,
                    },
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_user_memory",
            "description": (
                "查询用户的记忆信息（短期事实/长期偏好/最近交互摘要），"
                "用于推荐前结合用户历史偏好。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "想了解的偏好方向，如「品牌偏好」，可为空",
                    },
                },
                "required": [],
            },
        },
    },
]


# 工具名 → parameters 块（校验闸门的规则源）。
# 从 TOOL_DEFINITIONS 派生而非手写第二份，保证"给模型的说明书"与
# "执行层的闸门"永远同源——schema 改一次，校验自动跟进。
_SCHEMA_MAP: dict[str, dict] = {
    definition["function"]["name"]: definition["function"]["parameters"]
    for definition in TOOL_DEFINITIONS
}


# 同一签名的最大执行次数：第 3 次调用拦下。2 次里已含一次重试机会，因此不区分
# 成败（不设 retryable 字段）——规则只有"执行前计数，超上限即拒"这一条。
MAX_SAME_CALLS = 2


def _signature(name: str, arguments: dict) -> str:
    """调用签名 = 工具名 + 归一化参数的稳定序列化（键序无关）。

    只在参数通过契约校验之后调用，所以 `"10"` 与 `10` 是同一个签名——
    否则模型换个写法就绕过了熔断。
    """
    return f"{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"


def execute_tool(
    name: str, arguments: dict, seen: dict[str, int] | None = None,
) -> dict:
    """根据工具名称分发执行，返回统一信封 dict（执行层的唯一收口）。

    参数不合契约直接拒绝执行，返回可操作原因（模型改值重调）；
    同一签名本轮内超过 MAX_SAME_CALLS 次则熔断（`seen` 为 None 时不熔断）；
    漏网异常在此兜底；参数被归一化或夹紧时给信封追加 notice。
    """
    func = _TOOL_MAP.get(name)
    if not func:
        return fail(f"未知工具: {name}")
    if not isinstance(arguments, dict):
        return fail("工具参数必须是 JSON 对象")

    notes: list[str] = []
    schema = _SCHEMA_MAP.get(name)
    if schema:
        try:
            arguments, notes = validate_arguments(schema, arguments)
        except ArgumentError as exc:
            return fail(str(exc))

    # 熔断闸门：签名建在归一化后的参数上；拦下的那次不计入计数，
    # 因此到上限后每一次同签名调用都会被拦（不会"拦一次又放行"）
    if seen is not None:
        signature = _signature(name, arguments)
        if seen.get(signature, 0) >= MAX_SAME_CALLS:
            shown = signature if len(signature) <= 120 else signature[:117] + "..."
            print(f"[熔断] {shown} 本轮第 {MAX_SAME_CALLS + 1} 次，已拦下")
            return fail(
                f"本轮已用相同参数调用 {shown} 两次，请查看前面的工具消息"
                "（结果或失败原因都在那里），不要重复调用：换查询条件、换工具，"
                "或基于现有信息作答。"
            )
        seen[signature] = seen.get(signature, 0) + 1

    try:
        result = func(**arguments)
    except Exception as e:
        print(f"[工具异常] {name}: {e}")
        return fail(f"工具执行出错: {e}")

    if notes:
        result = {**result, "notice": "；".join(notes)}
    return result


def to_tool_message_content(result: dict) -> str:
    """把信封 dict 序列化为 tool 消息的 content（模型看到的是这段文本）。"""
    return json.dumps(result, ensure_ascii=False)
