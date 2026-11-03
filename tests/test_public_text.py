from app.api.public_text import public_assistant_text
from app.api.product_responses import public_compare_data


def test_public_assistant_text_hides_internal_product_id():
    text = "推荐苍岭雪脊 S5（LP-30），也可以比较 LP-26。"
    assert public_assistant_text(text) == "推荐苍岭雪脊 S5，也可以比较 候选商品。"


def test_public_assistant_text_keeps_price_after_removed_id():
    text = "极光熔岩猎影 16（LP-26，¥10,999）"
    assert public_assistant_text(text) == "极光熔岩猎影 16（¥10,999）"


def test_public_compare_data_uses_candidate_refs():
    result = public_compare_data({
        "products": [
            {"product_id": "LP-01", "name": "A"},
            {"product_id": "LP-02", "name": "B"},
        ],
        "fields": [{"field": "价格", "values": {"LP-01": 100, "LP-02": 200}}],
    })
    assert result["products"] == [
        {"name": "A", "candidate_no": 1},
        {"name": "B", "candidate_no": 2},
    ]
    assert result["fields"][0]["values"] == {"候选 1": 100, "候选 2": 200}
