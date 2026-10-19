"""请求内的过程事件；事件只用于展示，不阻塞 Agent 或数据库保存。"""

import asyncio


PHASES = {
    "requirements": "理解需求",
    "products": "筛选商品",
    "knowledge": "查询资料",
    "synthesis": "生成建议",
    "persist": "保存会话",
}

TOOL_PHASES = {
    "search_products": "products",
    "search_catalog": "products",
    "get_detail": "products",
    "compare_products": "products",
    "get_user_favorites": "products",
    "retrieve_knowledge": "knowledge",
    "load_skill": "requirements",
    "recall_user_memory": "requirements",
}

TOOL_TITLES = {
    "search_products": "语义筛选商品",
    "search_catalog": "按预算和条件筛选商品",
    "get_detail": "查询商品详情",
    "compare_products": "对比候选商品",
    "get_user_favorites": "读取收藏商品",
    "retrieve_knowledge": "检索选购资料",
    "load_skill": "准备处理流程",
    "recall_user_memory": "读取历史偏好",
}


def _phase_for(data: dict) -> str | None:
    if data.get("phase") in PHASES:
        return data["phase"]
    stage = str(data.get("stage", ""))
    if stage == "routing":
        return "requirements"
    if stage == "result":
        return "synthesis"
    if stage == "save":
        return "persist"
    if data.get("agent") == "presale":
        return "products"
    if data.get("agent") == "consult":
        return "knowledge"
    return TOOL_PHASES.get(data.get("tool"))


def emit(events: asyncio.Queue | None, event: str, **data) -> None:
    if events is None:
        return
    phase = _phase_for(data)
    if phase:
        data.setdefault("phase", phase)
        data.setdefault("title", PHASES[phase])
    tool = data.get("tool")
    if tool:
        data.setdefault("tool_title", TOOL_TITLES.get(tool, "执行查询"))
    # 慢客户端或断连不应卡住执行；队列满时丢弃最旧的展示事件。
    if events.full():
        events.get_nowait()
    events.put_nowait({"event": event, "data": data})
