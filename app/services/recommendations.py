"""从本轮 Agent 工具结果构造用户可见的推荐商品卡片。"""

from __future__ import annotations

import json
import re
from typing import Any

from app.services.product_refs import remember_product

_PRODUCT_ID = re.compile(r"(?<![A-Za-z0-9])(?:LP|PH|HP)-\d{2}(?!\d)")
_PRICE = re.compile(
    # 名称和价格之间允许出现的"填充物"：空白、表格竖线、括号、加粗符号，
    # 以及 Agent 偶尔写进括号的内部 ID（（LP-30）），它不是价格的一部分。
    r"(?:\s|[\|()（）\[\]*·—:：]|(?:LP|PH|HP)-\d{2}){0,12}"
    r"(?:[¥￥]\s*(?P<symbol>[\d,]+(?:\.\d+)?)"
    r"|(?P<unit>[\d,]+(?:\.\d+)?)\s*元)"
)
# 名称与价格之间允许的原文跨度（ID、括号、加粗会占字符，但不能跨句）。
_PRICE_GAP = 28


def _name_pattern(name: str) -> str:
    """容忍回复里的换行/空格断行：星海 潮汐 ANC ↔ 星海潮汐ANC。"""
    parts = [re.escape(part) for part in name.split()]
    return r"\s*".join(parts) if parts else ""


def _reply_price(reply: str, match: re.Match) -> float | None:
    """取商品名紧邻的金额；正文里写多少就展示多少，跨句的不算。"""
    window = reply[match.end(): match.end() + _PRICE_GAP]
    found = _PRICE.match(window)
    if found is None:
        return None
    raw = found.group("symbol") or found.group("unit") or ""
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _extract(messages: list[dict]) -> list[dict]:
    """收集工具结果里的商品；同一商品的多个视图按字段合并。

    对比工具只带 id + 名称，搜索工具才带价格/摘要——不能因为先看见
    缺字段的那份就把价格丢掉，所以这里是补空而非"先到先得"。
    """
    found: dict[str, dict] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content") or "{}")
        except (TypeError, ValueError):
            continue
        for value in _walk(payload.get("data") if isinstance(payload, dict) else None):
            product_id = value.get("product_id")
            name = value.get("name")
            if isinstance(product_id, str) and _PRODUCT_ID.fullmatch(product_id) and isinstance(name, str):
                entry = found.setdefault(product_id, {})
                for key, item in value.items():
                    if entry.get(key) in (None, "", []) and item not in (None, "", []):
                        entry[key] = item
    return list(found.values())


def build_recommendations(
    reply: str,
    messages: list[dict],
    *,
    user_id: str,
    session_id: str,
) -> list[dict]:
    """只返回最终回答中实际提到的商品，按回答中的出现顺序排列。

    价格以回答正文里写的为准（与用户看到的一致）；正文没写价才回退到
    工具结果的目录价。卡片不能出现"中间说 1499、右边写 1880"这种事。
    """
    candidates = []
    for product in _extract(messages):
        name = str(product.get("name", ""))
        pattern = _name_pattern(name)
        match = re.search(pattern, reply, flags=re.IGNORECASE) if pattern else None
        if match is not None:
            candidates.append((match.start(), product, match))
    candidates.sort(key=lambda item: item[0])

    cards = []
    for _, product, match in candidates[:8]:
        product_id = product["product_id"]
        name = str(product.get("name") or "商品")
        price = _reply_price(reply, match)
        ref, display_no = remember_product(user_id, session_id, product_id, name=name)
        cards.append({
            "product_id": product_id,
            "ref": ref,
            "display_no": display_no,
            "name": name,
            "brand": product.get("brand", ""),
            "category": product.get("category", "商品"),
            "price": price if price is not None else product.get("price"),
            "summary": product.get("summary") or (product.get("introduction") or [""])[0],
            "reason": "本轮回答中提到",
        })
    return cards


def merge_recommendations(
    existing: list[dict], incoming: list[dict], *, limit: int = 40,
) -> list[dict]:
    """本轮商品置顶，旧卡片保持原相对顺序；同一商品只保留最新一张。"""
    incoming_ids = {card.get("product_id") for card in incoming}
    rest = [card for card in existing if card.get("product_id") not in incoming_ids]
    return (incoming + rest)[:limit]


def public_cards(cards: list[dict]) -> list[dict]:
    """去掉内部 product_id：前端只认 ref，不接触目录编号。"""
    return [{key: value for key, value in card.items() if key != "product_id"}
            for card in cards]


def restore_cards(
    cards, *, user_id: str, session_id: str,
) -> list[dict]:
    """把落盘快照还原成可用卡片：重新签发 ref，编号沿用历史值。

    ref 只存在进程内存里，重启/换进程后旧 ref 必然失效，所以读取快照时
    必须重签；display_no 是用户可见编号，必须与历史保持一致。
    """
    restored = []
    for card in cards or ():
        product_id = card.get("product_id")
        if not product_id:
            continue
        ref, display_no = remember_product(
            user_id, session_id, str(product_id),
            name=str(card.get("name") or ""),
            display_no=int(card.get("display_no") or 0),
        )
        restored.append({**card, "ref": ref, "display_no": display_no})
    return restored
