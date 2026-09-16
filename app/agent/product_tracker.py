"""商品记忆（mentioned-products 独立存储）。

职责：跟踪对话中出现过的商品（ID / 名称 / 提及当时价格），供两处消费——
1. 注入：每次请求随摘要视图进入上下文，支撑跨压缩指代（"上次推荐那款"）
   与数值保真（模型可引用压缩前记录的价格，且字段名明确标注是历史价）；
2. 持久化：随会话落盘（storage v4），跨压缩累积，不随消息删除而丢失。

设计要点：
- 事实源唯一：名称与价格只从工具返回 JSON 提取（递归拆箱），
  对话文本与 tool_calls 参数里出现的 ID 只登记壳子；LLM 不参与本模块。
- 纯函数：merge_products 返回新 dict，不修改入参（与 compaction 同风格）。
- 提取时机：压缩事务内、删除前（delta 即将不可见时才需要记忆化）；
  窗口内尚可见的商品不必入店，避免重复注入。
"""

from __future__ import annotations

import json
import re
from typing import Any

_PRODUCT_ID_PATTERN = re.compile(r"\b(?:LP|PH|HP)-\d+\b")


def _record_product(products: dict, value: dict) -> None:
    """把一个含商品信息的字典登记进 products。

    products 结构：{商品ID: {"product_id": ..., "name": ..., "price_at_mention": ...}}
    - 首次见到该 ID：建立记录
    - 已经有记录：只补充/更新名称和价格
    """
    product_id = value.get("product_id")
    if not (isinstance(product_id, str) and _PRODUCT_ID_PATTERN.fullmatch(product_id)):
        return  # 没有 ID 或 ID 格式不对，不是可登记的商品
    record = products.setdefault(product_id, {"product_id": product_id})
    if isinstance(value.get("name"), str):
        record["name"] = value["name"]
    # 用 type() 而不是 isinstance()：True 也是 int 的实例，会把布尔值误当价格
    if type(value.get("price")) in (int, float):
        # 字段名显式标注"提及当时的价格"，防止下一轮 LLM 把记忆价当现价报给用户
        record["price_at_mention"] = value["price"]


def _collect_products(products: dict, data: Any) -> None:
    """遍历任意嵌套的列表/字典，把里面所有商品登记进 products。

    工具返回的 JSON 结构各不相同（商品可能藏在任意层级），
    所以用"待处理箱"逐层拆开检查，而不是写死查找路径。
    """
    pending = [data]                # 待处理箱：一开始只有整份数据
    while pending:
        current = pending.pop()     # 取出一件待检查的内容
        if isinstance(current, list):
            pending.extend(current)  # 列表：每件都放进待处理箱
        elif isinstance(current, dict):
            _record_product(products, current)
            pending.extend(current.values())  # 字典：先登记自己，再把它的值继续拆


def merge_products(store: dict, messages: list[dict]) -> dict:
    """把一段消息中出现的商品并入商品记忆，返回新 dict（不修改入参）。

    两个来源（可信度从低到高）：
    1. 对话文本与 tool_calls 参数里的商品 ID —— 只登记壳子，没有名称和价格
    2. 工具返回的 JSON —— 唯一可信的名称和价格来源
    """
    products: dict[str, dict] = {k: dict(v) for k, v in store.items()}

    # ---------- 来源一：对话文本与工具调用参数里的商品 ID ----------
    for message in messages:
        if message.get("role") == "tool":
            continue  # tool 消息交给来源二做结构化解析，这里只看人说的话
        text = message.get("content") or ""
        for product_id in _PRODUCT_ID_PATTERN.findall(text):
            products.setdefault(product_id, {"product_id": product_id})
        # 助手调用工具时，参数里也可能写了商品 ID（如 get_detail("LP-01")）
        for tc in message.get("tool_calls") or []:
            arguments = tc.get("function", {}).get("arguments") or ""
            for product_id in _PRODUCT_ID_PATTERN.findall(arguments):
                products.setdefault(product_id, {"product_id": product_id})

    # ---------- 来源二：工具返回的真实数据（唯一可信的名称/价格来源） ----------
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(message.get("content") or "{}")
        except (ValueError, TypeError):
            continue  # 内容不是合法 JSON，跳过
        if not isinstance(result, dict):
            continue
        data = result.get("data")
        if not isinstance(data, dict):
            continue
        _collect_products(products, data)
        # compare_products 的价格放在 fields 的特殊位置，与其他工具结构不同，单独补一遍
        for field in data.get("fields", []):
            if isinstance(field, dict) and field.get("field") == "价格":
                for product_id, price in field.get("values", {}).items():
                    if product_id in products and type(price) in (int, float):
                        products[product_id]["price_at_mention"] = price

    return products


def format_products_block(products: dict) -> str:
    """把商品记忆渲染成注入上下文的 <mentioned-products> 块；空店返回空串。"""
    if not products:
        return ""
    product_list = json.dumps(
        list(products.values()), ensure_ascii=False, separators=(",", ":"),
    )
    return f"<mentioned-products>\n{product_list}\n</mentioned-products>"
