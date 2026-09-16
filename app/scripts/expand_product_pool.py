"""扩容商品池：16 → 100 款（LLM 起草 + 程序校验，虚构纪律延续）。

目标：三品类各扩至 33±1 款（新增 84 款），刻意制造同质竞品（定位重叠），
让商品检索产生真实竞争密度——16 款下评测无区分度（Hit 恒 1.0 是天花板效应）。

防幻觉/防污染纪律（与评测集生成同源）：
- LLM 只起草字段内容，product_id 由脚本按品类前缀顺序分配（不信任 LLM 编号）
- 程序校验：7 字段 schema 精确匹配、specs 键模板与品类既有商品一致、
  价格区间、名称去重、真实品牌黑名单——任一失败重试或丢弃
- 虚构纪律：品牌限定现有虚构品牌池，型号名禁止出现真实品牌（黑名单校验）

产物：app/agent/tools/product_pool_extra.json
  ——由 mock_data.py 在导入时自动合并进 PRODUCTS / PRODUCT_INTRODUCTIONS

用法：
  python app/scripts/expand_product_pool.py                # 每品类补至 33/34 款
  python app/scripts/expand_product_pool.py --per-category 10   # 小批量试跑
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from app.db.snapshot import PRODUCTS  # noqa: E402
from app.config.settings import settings  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "agent" / "tools" / "product_pool_extra.json"

# 真实品牌黑名单：虚构纪律（2026-09-01 实测教训）——任何命中即拒绝
REAL_BRAND_BLOCKLIST = [
    "华为", "小米", "苹果", "OPPO", "vivo", "荣耀", "三星", "真我", "一加", "魅族",
    "联想", "戴尔", "惠普", "华硕", "宏碁", "微星", "神舟", "ThinkPad", "MacBook",
    "iPhone", "华为", "索尼", "Bose", "JBL", "森海塞尔", "铁三角", "Beats", "AirPods",
]

# 品类目标总量（含既有 16 款）
CATEGORY_TARGETS = {"笔记本": 34, "手机": 33, "耳机": 33}


def _category_spec_template(products: dict) -> dict[str, list[str]]:
    """从既有商品提取每品类的 specs 键模板（新商品必须严格一致，保证卡片渲染稳定）。"""
    templates: dict[str, list[str]] = {}
    for p in products.values():
        templates.setdefault(p["category"], list(p["specs"].keys()))
    return templates


def _compact_list(products: dict, category: str) -> str:
    """同品类现有商品紧凑列表（给 LLM 避免重复 + 允许竞品定位重叠）。"""
    return "\n".join(
        f"- {p['product_id']} {p['name']}（{p['price']} 元）"
        for pid, p in products.items() if p["category"] == category
    )


def _validate(prod: dict, category: str, spec_keys: list[str], lo: int, hi: int,
              existing_names: set[str], used_names: set[str], brands: list[str]) -> str | None:
    """程序校验，返回错误原因（None=通过）。任一失败即拒。"""
    if not isinstance(prod, dict):
        return "not a dict"
    for key in ("name", "brand", "category", "price", "specs", "introduction"):
        if key not in prod:
            return f"缺字段 {key}"
    if prod["category"] != category:
        return "品类不符"
    if prod["brand"] not in brands:
        return f"品牌越界: {prod['brand']}"
    if not isinstance(prod["price"], int) or not (lo <= prod["price"] <= hi):
        return f"价格越界: {prod['price']}"
    if not isinstance(prod["specs"], dict) or list(prod["specs"].keys()) != spec_keys:
        return "specs 键模板不符"
    if not (isinstance(prod["introduction"], list) and len(prod["introduction"]) == 2):
        return "introduction 必须 2 段"
    if any(not isinstance(x, str) or not (10 <= len(x) <= 90) for x in prod["introduction"]):
        return "introduction 段落长度不符"
    name = str(prod["name"])
    if name in existing_names or name in used_names:
        return f"名称重复: {name}"
    text = name + str(prod["brand"]) + "".join(prod["introduction"])
    for real in REAL_BRAND_BLOCKLIST:
        if real.lower() in text.lower():
            return f"真实品牌污染: {real}"
    return None


def main():
    parser = argparse.ArgumentParser(description="扩容商品池（LLM 起草 + 程序校验）")
    parser.add_argument("--per-category", type=int, default=0,
                        help="每品类新增数量，0=按 CATEGORY_TARGETS 补齐")
    parser.add_argument("--batch-size", type=int, default=2, help="每次 LLM 调用生成的商品数（大批次易 JSON 截断）")
    args = parser.parse_args()

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    model = settings.model_name

    brands = sorted({p["brand"] for p in PRODUCTS.values()})
    spec_templates = _category_spec_template(PRODUCTS)
    price_range = {"笔记本": (4000, 12000), "手机": (2000, 8000), "耳机": (300, 2500)}
    prefix = {"笔记本": "LP", "手机": "PH", "耳机": "HP"}

    existing_names = {p["name"] for p in PRODUCTS.values()}
    extra: dict[str, dict] = {}
    if OUT_PATH.exists():  # 断点续跑：已生成款直接复用（逐款增量写盘，崩溃安全）
        extra = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        for p in extra.values():
            existing_names.add(p["name"])
        print(f"续跑：已加载 {len(extra)} 款已生成扩容商品")

    def _save() -> None:
        OUT_PATH.write_text(
            json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"现有 {len(PRODUCTS)} 款，目标扩至 "
          f"{sum(CATEGORY_TARGETS.values())} 款（LLM 起草 + 程序校验）")

    for category, target in CATEGORY_TARGETS.items():
        cur = sum(1 for p in PRODUCTS.values() if p["category"] == category)
        cur_extra = sum(1 for p in extra.values() if p["category"] == category)
        need = (target - cur if not args.per_category else args.per_category) - cur_extra
        if need <= 0:
            print(f"\n[{category}] 已达标（{cur + cur_extra} 款），跳过")
            continue
        spec_keys = spec_templates[category]
        lo, hi = price_range[category]
        pfx = prefix[category]
        next_num = 1 + max(
            (int(pid.split("-")[1]) for pid in list(PRODUCTS) + list(extra)
             if pid.startswith(pfx + "-")),
            default=0,
        )

        print(f"\n[{category}] 现有 {cur + cur_extra}，还需 {need} 款（{pfx}-{next_num} 起，"
              f"specs 模板 {len(spec_keys)} 键）")
        accepted, rounds = 0, 0
        used_names: set[str] = set()
        while accepted < need and rounds < need * 4:
            rounds += 1
            k = min(args.batch_size, need - accepted)
            prompt = (
                f"你在为虚构电商「并夕夕」扩充商品目录。品类：{category}。\n"
                f"该品类现有商品（不要重复其定位与型号，可做竞品定位重叠）：\n"
                f"{_compact_list(PRODUCTS, category)}\n\n"
                f"约束：\n"
                f"- 品牌必须只用：{'、'.join(brands)}\n"
                f"- 型号名全部虚构，禁止出现任何真实品牌或真实型号\n"
                f"- 价格 {lo}-{hi} 元（int）\n"
                f"- specs 必须且只能用这些键：{spec_keys}，值要具体（如「6.7 英寸 2K 120Hz」）\n"
                f"- introduction 恰好 2 段，每段 20~60 字\n"
                f"- 定位可与现有商品重叠（形成竞品），也可填补空缺定位\n\n"
                f"请创作 {k} 款新的{category}商品。"
                f'输出 JSON：{{"products": [{{"name": "...", "brand": "...", '
                f'"category": "{category}", "price": 0, "specs": {{}}, '
                f'"introduction": ["...", "..."]}}]}}'
            )
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=6144,
                    reasoning_effort="low",
                    timeout=60,
                )
                raw = resp.choices[0].message.content.strip()
                import re
                m = re.search(r"\{.*\}", raw, re.S)
                payload = json.loads(m.group(0) if m else raw)
                new_items = payload.get("products", [])
            except Exception as exc:  # noqa: BLE001
                print(f"   [warn] 批次 {rounds} LLM 失败: {type(exc).__name__}: {exc}")
                time.sleep(2)
                continue

            for item in new_items:
                if accepted >= need:
                    break
                err = _validate(item, category, spec_keys, lo, hi,
                                existing_names, used_names, brands)
                if err:
                    print(f"   [reject] {item.get('name', '?')}: {err}")
                    continue
                pid = f"{pfx}-{next_num:02d}"
                next_num += 1
                item["product_id"] = pid
                used_names.add(item["name"])
                extra[pid] = item
                accepted += 1
                _save()  # 逐款增量写盘：超时/崩溃不丢已生成款（首轮教训）
                print(f"   + {pid} {item['name']}（{item['price']} 元）")
            time.sleep(0.8)  # 调用间隔防限流
        if accepted < need:
            print(f"   [warn] {category} 仅完成 {accepted}/{need}")

    _save()
    print(f"\n🎉 扩容完成：新增 {len(extra)} 款 → {OUT_PATH}")
    print("   下一步：重建 product_kb（build_product_kb.py 会自动合并扩容款）")


if __name__ == "__main__":
    main()
