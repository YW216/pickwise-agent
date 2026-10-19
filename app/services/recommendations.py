"""从本轮 Agent 工具结果构造用户可见的推荐商品卡片。"""

from __future__ import annotations

import json
import re
from typing import Any

from app.db.catalog_repo import alist_product_cards
from app.services.product_images import category_image, product_image
from app.services.product_refs import remember_product

_PRODUCT_ID = re.compile(r"(?<![A-Za-z0-9])(?:LP|PH|HP)-\d{2}(?!\d)")
# Markdown 表格行：整行以 | 开头和结尾，商品名与参数在同一行。
_BULLET = re.compile(r"^[-*+]\s|^\d+[.、)]\s")
# 主推语气词；后面紧跟的商品就是「首选」。
_BEST_HINT = re.compile(r"(?:首选|最推荐|推荐选|建议选|优先考虑)([^\n]{0,60})")
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
    """容忍回复里的换行/空格断行：星海 潮汐 ANC ↔ 星海潮汐ANC。

    末尾断言「后面不能接空格 + 字母数字」：避免短名吃掉长名前缀——
    「云章 轻羽 Air」不能匹配到「云章 轻羽 Air 13」。
    """
    parts = [re.escape(part) for part in name.split()]
    if not parts:
        return ""
    return r"\s*".join(parts) + r"(?!\s*[0-9A-Za-z])"


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


def _catalog_matches(scope: str, known: dict, catalog: list[dict]) -> dict:
    """在回复文本里没命中工具结果、但确实存在的目录商品。

    名字长度降序匹配，避免短名抢先吃掉长名的前缀（如「雪脊 S5」与「雪脊 S5 16GB」）。
    """
    extra: dict[str, dict] = {}
    for product in sorted(catalog, key=lambda item: len(item["name"]), reverse=True):
        if product["product_id"] in known:
            continue
        pattern = _name_pattern(product["name"])
        if pattern and re.search(pattern, scope, flags=re.IGNORECASE):
            extra[product["product_id"]] = product
    return extra


def _display_scope(reply: str, catalog: list[dict]) -> str:
    """挑出"陈列行"：表格、标题、列表项；非表格行里点数多款的视为罗列并剔除。"""
    if not catalog:
        return reply
    patterns = [
        (product["product_id"], _name_pattern(product["name"]))
        for product in catalog
    ]
    kept: list[str] = []
    for raw in reply.splitlines():
        line = raw.strip()
        if not line:
            continue
        is_table = line.startswith("|")
        if not (is_table or line.startswith("#") or _BULLET.match(line)):
            continue
        if not is_table:
            hits = sum(1 for _, pattern in patterns
                       if pattern and re.search(pattern, line, flags=re.IGNORECASE))
            if hits > 1:
                continue
        kept.append(line)
    return "\n".join(kept)


def _best_index(reply: str, candidates: list) -> int:
    """主推商品：回复里「首选/最推荐/建议选」后面跟的那款；没有就取第一张。"""
    for hint in _BEST_HINT.finditer(reply):
        window = hint.group(1)
        for index, (_, product, _) in enumerate(candidates):
            pattern = _name_pattern(str(product.get("name", "")))
            if pattern and re.search(pattern, window, flags=re.IGNORECASE):
                return index
    return 0


async def build_recommendations(
    reply: str,
    messages: list[dict],
    *,
    user_id: str,
    session_id: str,
    database_url: str | None = None,
) -> list[dict]:
    """只返回最终回答里"真正展示过"的商品，按出现顺序排列。

    展示口径：表格行、标题行、列表项才算陈列；正文里顺口罗列的段落不算。
    一行里同时点名多款商品（"还有其他可选：A、B、C"）视为罗列，整行跳过——
    这类商品用户可以点开列表看，但不该挤进右侧。表格行例外：多栏清单
    本身就是陈列，允许一行出现多款。

    价格以回答正文里写的为准（与用户看到的一致）；正文没写价才回退到
    工具结果的目录价。卡片不能出现"中间说 1499、右边写 1880"这种事。
    """
    catalog: list[dict] = []
    if database_url:
        try:
            catalog = await alist_product_cards(database_url)
        except Exception:  # 目录不可用时退回工具结果，不影响主流程
            catalog = []
    scope = _display_scope(reply, catalog) or reply
    found = {product["product_id"]: product for product in _extract(messages)}
    if catalog:
        # Agent 凭商品记忆作答时，本轮可能没有对应工具结果；按名字回查目录补齐。
        found.update(_catalog_matches(scope, found, catalog))
    candidates = []
    for product in found.values():
        name = str(product.get("name", ""))
        pattern = _name_pattern(name)
        match = re.search(pattern, scope, flags=re.IGNORECASE) if pattern else None
        if match is not None:
            candidates.append((match.start(), product, match))
    candidates.sort(key=lambda item: item[0])

    cards = []
    best_index = _best_index(reply, candidates)
    # 不设条数上限：中间展示了几款，右侧就要有几款（同一份展示口径）。
    for index, (_, product, match) in enumerate(candidates):
        product_id = product["product_id"]
        name = str(product.get("name") or "商品")
        price = _reply_price(scope, match)
        ref, display_no = remember_product(user_id, session_id, product_id, name=name)
        cards.append({
            "product_id": product_id,
            "ref": ref,
            "display_no": display_no,
            "name": name,
            "brand": product.get("brand", ""),
            "category": product.get("category", "商品"),
            "image": product_image(product),
            "price": price if price is not None else product.get("price"),
            "summary": product.get("summary") or (product.get("introduction") or [""])[0],
            "reason": "本轮回答中提到",
            # 首行商品即 Agent 的主推，卡片上用「首选」标出来（只保留一个）。
            "best": index == best_index,
        })
    return cards


def _name_key(name) -> str:
    return re.sub(r"\s+", "", str(name or "")).lower()


def merge_recommendations(
    existing: list[dict], incoming: list[dict], *, limit: int = 40,
) -> list[dict]:
    """本轮商品置顶，旧卡片保持原相对顺序；同一商品只保留最新一张。

    去重同时看 product_id 和商品名：目录里若存在两条同名记录（不同 ID），
    用户看到的仍是同一款，右侧不该出现两张一样的卡片。
    """
    merged: list[dict] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for card in list(incoming) + list(existing):
        product_id = card.get("product_id")
        name = _name_key(card.get("name"))
        if (product_id and product_id in seen_ids) or (name and name in seen_names):
            continue
        if product_id:
            seen_ids.add(product_id)
        if name:
            seen_names.add(name)
        merged.append(card)
        if len(merged) >= limit:
            break
    # 「首选」全局只有一个：本轮第一张优先，历史卡片上的标记让位。
    leader = next((card for card in merged if card.get("best")), None)
    for card in merged:
        card["best"] = card is leader
    return merged


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
        restored.append({
            **card,
            "ref": ref,
            "display_no": display_no,
            "image": card.get("image") or category_image(str(card.get("category") or "")),
        })
    return restored
