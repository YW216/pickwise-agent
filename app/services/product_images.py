"""商品示意图：品类级图片 → 稳定 URL。

当前商品池没有真实素材，先按品类共用一张图（assets/products/category）。
后续要给单个商品换专属图时，只需在这里按 product_id 优先匹配，
调用方（卡片/详情）拿到的仍是同一个字段，前端不用改。
"""

from __future__ import annotations

from pathlib import Path

MEDIA_ROOT = Path(__file__).resolve().parents[2] / "assets" / "products"

_CATEGORY_SLUGS = {
    "耳机": "earbuds",
    "笔记本": "laptop",
    "手机": "phone",
    "平板": "tablet",
}


def category_image(category: str | None) -> str | None:
    """中文品类 → 本地图片 URL；未知品类返回 None，前端回退纯文字卡片。"""
    slug = _CATEGORY_SLUGS.get((category or "").strip())
    if slug is None:
        return None
    if not (MEDIA_ROOT / "category" / f"{slug}.webp").exists():
        return None
    return f"/media/category/{slug}.webp"


def product_image(product: dict) -> str | None:
    """单个商品的图片：优先专属图，没有就落到品类图。"""
    product_id = product.get("product_id")
    if product_id:
        for suffix in ("webp", "jpg", "png"):
            if (MEDIA_ROOT / f"{product_id}.{suffix}").exists():
                return f"/media/{product_id}.{suffix}"
    return category_image(str(product.get("category") or ""))
