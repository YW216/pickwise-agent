"""商品和收藏共用的 HTTP 响应转换。"""

from __future__ import annotations

from app.api.envelope import error, success
from app.services.product_images import product_image


def detail_response(result: dict, request_id: str):
    if result["success"]:
        data = dict(result["data"])
        product = data.get("product")
        if isinstance(product, dict):
            image = product_image(product)
            data["product"] = {
                key: value for key, value in product.items()
                if key != "product_id"
            }
            if image:
                data["product"]["image"] = image
        return success(data, request_id)
    if result.get("data", {}).get("unavailable"):
        return error(503, "PRODUCT_SERVICE_UNAVAILABLE",
                     "商品服务暂时不可用", request_id)
    return error(404, "PRODUCT_NOT_FOUND", "商品不存在", request_id)


def product_response(result: dict, request_id: str):
    """商品搜索工具信封 → HTTP（§4.5：按 data 内稳定标志判断，不匹配文案）。"""
    if result["success"]:
        return success(public_product_data(result["data"]), request_id)
    data = result.get("data", {})
    if data.get("unavailable"):
        return error(503, "PRODUCT_SERVICE_UNAVAILABLE",
                     "商品服务暂时不可用", request_id)
    # 无匹配（含 nearest 兜底）是 200 + 空结果：matched=0 与参考标记原样透传
    return success(public_product_data(data), request_id)


def public_product_data(data: dict) -> dict:
    """将内部商品卡片转换为用户展示卡片。

    product_id 只在 Agent/数据库内部流转；HTTP 返回候选序号，避免把
    内部目录主键暴露给浏览器。工具层仍使用原始数据，不影响详情查询。
    """
    payload = dict(data or {})
    for key in ("results", "candidates", "nearest"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        public_items = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                public_items.append(item)
                continue
            public_item = dict(item)
            image = product_image(item)
            public_item.pop("product_id", None)
            public_item["candidate_no"] = index
            if image:
                public_item["image"] = image
            public_items.append(public_item)
        payload[key] = public_items
    return payload


def public_compare_data(data: dict) -> dict:
    """隐藏 compare 响应中的主键，并用候选序号对齐字段值。"""
    payload = dict(data or {})
    products = payload.get("products")
    if not isinstance(products, list):
        return payload
    refs: dict[str, str] = {}
    public_products = []
    for index, item in enumerate(products, start=1):
        if not isinstance(item, dict):
            public_products.append(item)
            continue
        product_id = item.get("product_id")
        if isinstance(product_id, str):
            refs[product_id] = f"候选 {index}"
        public_item = {
            key: value for key, value in item.items()
            if key != "product_id"
        }
        public_item["candidate_no"] = index
        public_products.append(public_item)
    payload["products"] = public_products
    fields = payload.get("fields")
    if isinstance(fields, list):
        public_fields = []
        for field in fields:
            if not isinstance(field, dict):
                public_fields.append(field)
                continue
            public_field = dict(field)
            values = field.get("values")
            if isinstance(values, dict):
                public_field["values"] = {
                    refs.get(key, "候选商品"): value
                    for key, value in values.items()
                }
            public_fields.append(public_field)
        payload["fields"] = public_fields
    return payload
