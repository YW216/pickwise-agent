"""工具统一返回信封（PickWise）。

所有工具返回**同一个 dict 结构**——`ok()`/`fail()` 直接产出 dict，不包 dataclass：
信封最终要 `json.dumps` 进 tool 消息，中间夹一层类只是白转一次。失败路径统一：

- ``ok(data)``   —— 成功：success=True，data 放业务数据
- ``fail(msg)``  —— 失败：success=False，error 放**纯人话**文案

设计要点：
1. 工具返回的是 dict（信封），不是异常——失败对模型是"需要看到的输入"，
   不是"需要避开的意外"（模型据此决定下一步：换条件 / 如实告知用户）。
2. error 只放可读文案，不拼异常细节（模型可能转述给用户）。
   例外：执行器层（registry.execute_tool）兜底的未知异常会拼异常信息，
   供开发者排查（并在终端 print）。
3. 流向：模型看序列化后的 JSON 文本（tool 消息 content），
   程序看对象（result["data"]），评估看插桩记录。
4. 唯一例外：执行器层（registry.execute_tool）在参数被归一化或夹紧时，为信封
   追加 notice 字段（人话提示）。它描述"这次调用的参数被调整过"，不属于业务
   数据故不进 data——ok()/fail() 本身不产生该字段。
"""

from typing import Optional


def ok(data: dict) -> dict:
    """成功返回。

    Args:
        data：业务数据（如 ``{"candidates": [...], "total": 3}``）

    Returns:
        信封 dict，success=True
    """
    return {"success": True, "error": None, "data": data or {}}


def fail(msg: str, data: Optional[dict] = None) -> dict:
    """失败返回（业务查无结果 / 异常兜底）。

    Args:
        msg：失败原因，**纯人话**（如"没有符合条件的商品，请调整预算或条件"）
        data：可选的业务数据（查无结果时通常给空列表，便于模型理解）

    Returns:
        信封 dict，success=False
    """
    return {"success": False, "error": msg, "data": data or {}}
