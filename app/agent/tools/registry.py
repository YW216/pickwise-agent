"""工具注册表：OpenAI function calling schema + 分发执行（PickWise）。

结构仿 ecom：
- TOOL_DEFINITIONS：给模型看的工具说明书（手写 OpenAI function schema）
- _TOOL_MAP       ：工具名 → 业务函数
- execute_tool    ：统一分发 + 执行器层兜底（第二层防御）

错误识别：
- 工具函数内（第一层）负责业务失败的人话文案（"商品不存在"）
- execute_tool（第二层）兜底未知异常：终端 print 排查 + 返回拼异常的 error

schema 编写约定：

1. 体积：每轮请求都会重发 schema，体积直接影响 token 成本。
   description 只写「做什么 + 返回什么 + 参数怎么填」，场景示例一律不放。

2. 边界（重要）：可以写「何时用我」和「我不是什么」这类**自我界定**，
   但**禁止写「去用别的工具」「别用 A 代替 B」这类跨工具路由**。
   原因：同一工具会被多个 Agent 共享而白名单不同，指向别的工具的指令
   ——对没有那个工具的 Agent 是噪音，更糟时会诱导它调用看不见的工具名。
   路由规则统一写在 app/prompts/agents.py 各 Agent 的「工具使用原则」里。

3. 取值固定的参数用 enum 表达，既省字数又带一层值域约束。
"""

import json
from typing import Callable

from app.agent.tools.catalog import search_catalog
from app.agent.tools.detail import compare_products, get_detail
from app.agent.tools.favorites import get_user_favorites
from app.agent.tools.knowledge import retrieve_knowledge
from app.agent.tools.memory_tool import recall_user_memory
from app.agent.tools.search_products import search_products
from app.agent.tools.skill_tool import load_skill

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
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回条数，默认 10",
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
                    },
                    "category": {
                        "type": "string",
                        "description": "品类过滤",
                        "enum": ["笔记本", "手机", "耳机"],
                    },
                    "max_price": {
                        "type": "integer",
                        "description": "预算上限（元），如 5000 表示只要 5000 元及以下的商品",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回商品数上限，默认 5",
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
            "description": "按 product_id 查单个商品的价格与完整参数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "商品 ID，如 LP-01",
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
                        "items": {"type": "string"},
                        "description": "商品 ID 列表，2-4 个，如 ['LP-01','LP-02']",
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
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回片段数，默认 3",
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


def execute_tool(name: str, arguments: dict) -> dict:
    """根据工具名称分发执行，返回统一信封 dict。

    第二层防御：捕获任何未被工具函数处理的异常，
    终端 print 供排查，返回拼异常的 error（纯人话优先由工具函数负责）。
    """
    func = _TOOL_MAP.get(name)
    if not func:
        return {"success": False, "error": f"未知工具: {name}", "data": {}}

    try:
        return func(**arguments)
    except Exception as e:
        print(f"[工具异常] {name}: {e}")
        return {"success": False, "error": f"工具执行出错: {e}", "data": {}}


def to_tool_message_content(result: dict) -> str:
    """把信封 dict 序列化为 tool 消息的 content（模型看到的是这段文本）。"""
    return json.dumps(result, ensure_ascii=False)
