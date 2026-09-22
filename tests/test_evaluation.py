"""评估模块单元测试（P6）：全部离线，不调 LLM。

覆盖：
- 用例集加载与字段映射（20 条，分类齐全）
- 规则判分库的每类断言（用合成 RunTrace，正反例都要有）
- 判分器自检路径（编造数据必须被查出）
- 沙箱的隔离性与插桩还原（mock client，不真调）

端到端真调验证由 app/scripts/run_eval.py 承担（--self-test + 小跑 + 全量）。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation import metrics
from app.evaluation.dataset import EvalCase, load_dataset
from app.evaluation.trace import RunTrace, ToolObservation
from app.db.snapshot import PRODUCTS

ROOT = Path(__file__).resolve().parent.parent
_OK = 0
_FAIL = 0


def _ok(msg: str):
    global _OK
    _OK += 1
    print(f"  ✅ {msg}")


def _fail(msg: str):
    global _FAIL
    _FAIL += 1
    print(f"  ❌ {msg}")


def _check(name: str, case: EvalCase, trace: RunTrace):
    """对合成轨迹执行单一断言，返回其 CheckResult。"""
    return getattr(metrics, f"check_{name}")(case, trace)


def _trace(replies, routes=None, tools=None, failed_tools=None):
    """合成轨迹。tools=成功调用（信封 success=true）；failed_tools=失败调用。"""
    obs = [ToolObservation(name=n, arguments={}, result='{"success": true, "data": {}}')
           for n in (tools or [])]
    obs += [ToolObservation(name=n, arguments={}, result='{"success": false, "error": "没有找到"}')
            for n in (failed_tools or [])]
    return RunTrace(
        case_id="t", turns=["x"] * len(replies),
        replies=list(replies),
        routes=routes if routes is not None else [["presale"]] * len(replies),
        tool_observations=obs,
    )


def test_dataset_load():
    print("\n[1/5] 用例集加载")
    cases = load_dataset(ROOT / "app" / "evaluation" / "cases.json")
    if len(cases) == 20:
        _ok("20 条用例")
    else:
        _fail(f"用例数 {len(cases)} != 20")
    categories = {c.category for c in cases}
    expected = {"主流程", "对比", "咨询", "跨轮指代", "边界", "混合"}
    if categories == expected:
        _ok(f"分类齐全: {sorted(categories)}")
    else:
        _fail(f"分类异常: {categories}")
    if all(c.expected_route is not None for c in cases if c.category != "边界"):
        _ok("非边界用例均带路由归因断言")
    else:
        _fail("存在缺路由断言的非边界用例")
    if not any(c.category == "路由" for c in cases):
        _ok("无纯路由类目（分层裁定：路由专测归 router_cases.json）")
    else:
        _fail("仍存在纯路由类目")


def test_rule_price_checks():
    print("\n[2/5] 数值保真断言")
    case = EvalCase(
        id="t", category="主流程", description="t", turns=["x"],
        expected_tools=["search_catalog"],
        reply_prices_within={"category": "耳机", "max_price": 1500},
        must_include_catalog_price={"HP-01": 1260},
        valid_ids_only=True,
    )
    good = _trace(["推荐 星海 凌霄 Buds Pro（HP-01），售价 ¥1260，适合通勤降噪"],
                  tools=["search_catalog"])
    for name in ("valid_ids", "prices_within", "must_include", "tools"):
        r = _check(name, case, good)
        ( _ok if r.passed else _fail)(f"正例通过 {name}: {r.detail}")

    bad = _trace(["推荐 幻影 X9（LP-99），售价 ¥9999"], tools=["search_catalog"])
    for name in ("valid_ids", "must_include"):
        r = _check(name, case, bad)
        ( _ok if r.applicable and not r.passed else _fail)(f"编造数据被查出 {name}: {r.detail}")

    # 目录价超预算 → 失败（预算遵守只看目录价的最小值）
    over = _trace(["推荐 曜石 磐石 Studio（HP-02），售价 ¥2940"], tools=["search_catalog"])
    r = _check("prices_within", case, over)
    ( _ok if not r.passed else _fail)(f"目录价超预算被拒: {r.detail}")
    # 非目录价数字（差价/省了多少钱）不判违规
    derived = _trace(["HP-01 售价 ¥1260，比 HP-05 便宜 280 元"], tools=["search_catalog"])
    r = _check("prices_within", case, derived)
    ( _ok if r.passed else _fail)(f"差价数字不误报: {r.detail}")

    # 断言未声明 → applicable=False（不计分）
    r = _check("must_include", EvalCase(id="t", category="x", description="x", turns=["x"]), _trace(["任意"]))
    ( _ok if not r.applicable else _fail)("未声明的断言跳过计分")


def test_id_pattern_matches_id_adjacent_to_chinese():
    """回复里的商品 ID 紧贴中文时也要能提取（单词边界在中文旁不成立）。"""
    assert metrics._ID_PATTERN.findall("推荐LP-01这款") == ["LP-01"]
    assert metrics._ID_PATTERN.findall("价格LP-03为5999") == ["LP-03"]
    assert metrics._ID_PATTERN.findall("推荐 LP-05 这款") == ["LP-05"]
    assert metrics._ID_PATTERN.findall("XLP-01 和 LP-011") == []


def test_rule_content_checks():
    print("\n[3/5] 内容与边界断言")
    # 关键词
    case = EvalCase(id="t", category="咨询", description="t", turns=["x"],
                    keyword_hits=["12 个月", "80%"])
    r = _check("keywords", case, _trace(["耳机整机保修 12 个月，电池健康度低于 80% 可免费更换。"]))
    ( _ok if r.passed else _fail)(f"关键词全命中: {r.detail}")
    r = _check("keywords", case, _trace(["保修一年。"]))
    ( _ok if r.applicable and not r.passed else _fail)(f"缺关键词被查出: {r.detail}")

    # 空工具约束（闲聊）
    case = EvalCase(id="t", category="边界", description="t", turns=["x"], expected_tools=[])
    r = _check("tools", case, _trace(["你好！"], tools=[]))
    ( _ok if r.passed else _fail)("闲聊零工具通过")
    r = _check("tools", case, _trace(["你好！"], tools=["search_catalog"]))
    ( _ok if not r.passed else _fail)("闲聊调工具被查出")

    # any-of 工具约束（咨询：knowledge 或 skill 二选一；warranty 工具已下线）
    case = EvalCase(id="t", category="咨询", description="t", turns=["x"],
                    expected_tools=["retrieve_knowledge", "load_skill"])
    r = _check("tools", case, _trace(["……"], tools=["load_skill"]))
    ( _ok if r.passed else _fail)("any-of 命中其一通过")

    # 路由归因
    case = EvalCase(id="t", category="主流程", description="t", turns=["a", "b"],
                    expected_route=["presale", "consult"])
    r = _check("route", case, _trace(["a", "b"], routes=[["presale"], ["consult"]]))
    ( _ok if r.passed else _fail)("多轮路由一致通过")
    r = _check("route", case, _trace(["a", "b"], routes=[["presale"], ["presale"]]))
    ( _ok if not r.passed else _fail)("第二轮路由错误被查出")

    # last_turn_only（改预算场景：第一轮超新预算不算失败）
    case = EvalCase(id="t", category="跨轮指代", description="t", turns=["a", "b"],
                    reply_prices_within={"category": "笔记本", "max_price": 4500,
                                         "last_turn_only": True})
    r = _check("prices_within", case, _trace(["LP-01 售价 ¥5180", "LP-03 售价 ¥3960"]))
    ( _ok if r.passed else _fail)(f"last_turn_only 只查末轮: {r.detail}")

    # 失败调用不算命中 expected_tools（recommend_reviews 教训：0 结果的错误调用≠信息需求被满足）
    case = EvalCase(id="t", category="主流程", description="t", turns=["x"],
                    expected_tools=["search_products"])
    r = _check("tools", case, _trace(["……"], failed_tools=["search_products"]))
    ( _ok if not r.passed else _fail)(f"仅失败调用不命中: {r.detail}")
    r = _check("tools", case, _trace(["……"], tools=["search_products"]))
    ( _ok if r.passed else _fail)("成功调用命中")

    # tools_success_required=False：查无结果类用例按"发起过调用"判定（no_match_honesty 教训）
    case = EvalCase(id="t", category="边界", description="t", turns=["x"],
                    expected_tools=["search_catalog", "search_products"],
                    tools_success_required=False)
    r = _check("tools", case, _trace(["……"], failed_tools=["search_products"]))
    ( _ok if r.passed else _fail)(f"查无结果类：失败调用也算命中: {r.detail}")
    r = _check("tools", case, _trace(["……"]))
    ( _ok if not r.passed else _fail)(f"查无结果类：完全没调用仍不通过: {r.detail}")

    # result 为 dict（沙箱采集的真实形态）与 JSON 串（序列化形态）都算命中
    dict_obs = ToolObservation(name="search_products", arguments={},
                               result={"success": True, "data": {}})
    t = RunTrace(case_id="t", turns=["x"], replies=["……"],
                 routes=[["presale"]], tool_observations=[dict_obs])
    case = EvalCase(id="t", category="主流程", description="t", turns=["x"],
                    expected_tools=["search_products"])
    r = metrics.check_tools(case, t)
    ( _ok if r.passed else _fail)(f"dict 形态结果命中: {r.detail}")

    # prices_in_catalog：预算回显豁免，真编造仍被查
    case = EvalCase(id="t", category="边界", description="t", turns=["300 块以内有降噪耳机吗"],
                    reply_prices_in_catalog=True)
    echo = RunTrace(case_id="t", turns=case.turns,
                    replies=["300 元以内暂无符合条件的降噪耳机。"])
    r = metrics.check_prices_in_catalog(case, echo)
    ( _ok if r.passed else _fail)(f"预算回显不误报: {r.detail}")
    # 编造价格取目录最高价 +1——固定数字会随商品池扩容变成真实目录价（5999 的教训）
    non_catalog_price = max(p["price"] for p in PRODUCTS.values()) + 1
    r = _check("prices_in_catalog", case, _trace([f"推荐 幻影 Pro，只要 ¥{non_catalog_price}"]))
    ( _ok if not r.passed else _fail)(f"编造价格被查出: {r.detail}")

    # must_include_any（多轮指代宽松版）
    case = EvalCase(id="t", category="跨轮指代", description="t", turns=["a", "b"],
                    must_include_catalog_price_any={"HP-01": 1260, "HP-04": 1880})
    r = _check("must_include_any", case, _trace(["推荐 HP-04 ¥1880", "HP-04 是 1880 元"]))
    ( _ok if r.passed else _fail)("must_include_any 命中一对通过")


def test_self_test_grader():
    print("\n[4/5] 判分器自检入口")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_eval", ROOT / "app" / "scripts" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod._self_test():
        _ok("run_eval --self-test 逻辑通过（编造数据被查出）")
    else:
        _fail("run_eval --self-test 判分器漏检")


def test_judge_parser():
    print("\n[5/6] LLM judge 解析器（离线，不真调）")
    from app.evaluation.judges import _parse_judge_response

    r = _parse_judge_response('{\"score\": 4, \"reasons\": [\"结构清晰\", \"理由有据\"]}')
    ( _ok if r == {"score": 4, "reasons": ["结构清晰", "理由有据"]} else _fail)("合法 JSON 解析")

    r = _parse_judge_response('评审结论如下：{"score": 3, "reasons": ["尚可"]}，请参考。')
    ( _ok if r and r["score"] == 3 else _fail)("包裹在散文中的 JSON 也能提取")

    r = _parse_judge_response('{\"score\": 9, \"reasons\": []}')
    ( _ok if r is None else _fail)("分数越界拒绝")
    r = _parse_judge_response('{\"score\": \"4\", \"reasons\": []}')
    ( _ok if r is None else _fail)("字符串分数拒绝（防 bool/str 混入）")
    r = _parse_judge_response('我觉得回答不错，给个好评！')
    ( _ok if r is None else _fail)("无 JSON 输出拒绝")
    r = _parse_judge_response('{\"score\": 4}')
    ( _ok if r == {"score": 4, "reasons": []} else _fail)("reasons 缺失时降级为空列表（不判失败）")


def test_report_aggregation():
    print("\n[5/5] 聚合报告")
    from app.evaluation.evaluator import CaseResult, EvalReport
    report = EvalReport(results=[
        CaseResult(case_id="a", category="主流程", description=""),
        CaseResult(case_id="b", category="主流程", description=""),
        CaseResult(case_id="c", category="边界", description="",
                   error="TimeoutError"),
    ])
    if report.passed_count == 2:
        _ok("error 用例判不通过、正常用例默认通过")
    else:
        _fail(f"聚合异常: {report.passed_count}")
    summary = report.to_dict()["summary"]
    if summary["by_category"]["主流程"] == {"passed": 2, "total": 2}:
        _ok("分类汇总正确")
    else:
        _fail(f"分类汇总异常: {summary['by_category']}")


def main():
    print("=" * 72)
    print("  评估模块单元测试（离线，无 LLM 调用）")
    print("=" * 72)
    test_dataset_load()
    test_rule_price_checks()
    test_rule_content_checks()
    test_self_test_grader()
    test_judge_parser()
    test_report_aggregation()
    print("\n" + "=" * 72)
    if _FAIL == 0:
        print(f"  🎉 全部通过（{_OK} 项断言）")
        sys.exit(0)
    print(f"  ❌ {_FAIL} 项失败 / {_OK} 项通过")
    sys.exit(1)


if __name__ == "__main__":
    main()
