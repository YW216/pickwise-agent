"""工具参数的两道闸门：协议层（JSON 解析）与契约层（schema 校验）。

规则源是 TOOL_DEFINITIONS[].parameters 本身——schema 是给模型的说明书，
反读它做校验，就不存在"说明书改了、闸门忘改"。

三类处置，判据是错误有没有消解歧义：
- 无损归一化（静默）："10" → 10；可选参数给 null 视为未提供；
- 边界夹紧（修 + 记入 notes）：limit 超上限、数组超长取前 N 项；
- 语义拒绝（抛 ArgumentError）：enum 越界、必填缺失、数组不足下限。
"""

import json


class ArgumentError(ValueError):
    """参数不合工具契约。文案面向模型，会被回喂要求改值。"""


def parse_tool_arguments(raw: str | None) -> tuple[dict | None, str | None]:
    """协议层：模型给的 arguments 字符串 → 参数字典。

    刻意不修补坏 JSON——截断若发生在值中间，补出来的是错参数且完全静默。
    返回 (参数, None) 表示通过；(None, 回喂文案) 表示需模型重发。
    """
    text = (raw or "").strip()
    if not text:
        return None, "工具参数为空：请重新调用并给出完整参数（JSON 对象）。"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"工具参数不是合法 JSON（{exc.msg}，位置 {exc.pos}）：请重新输出 JSON 对象。"
    if not isinstance(parsed, dict):
        return None, f"工具参数必须是 JSON 对象，收到 {type(parsed).__name__}：请重新调用。"
    return parsed, None


def validate_arguments(schema: dict, arguments: dict) -> tuple[dict, list[str]]:
    """契约层：按 schema 校验并归一化，返回 (参数, 调整说明)。

    说明非空表示参数被归一化或夹紧（调用方须随结果告知模型）；
    存在不可无损纠正的问题时抛 ArgumentError。
    """
    properties = schema.get("properties") or {}
    missing = [k for k in schema.get("required") or [] if arguments.get(k) is None]
    if missing:
        raise ArgumentError(f"缺少必需参数：{'、'.join(missing)}。请补齐后重新调用。")

    normalized: dict = {}
    notes: list[str] = []
    for name, value in arguments.items():
        spec = properties.get(name)
        if spec is None:  # 未知参数：过滤而非拒绝（拒绝白烧一轮），但须告知正确参数集
            notes.append(f"已忽略未知参数 {name}（可用参数：{'、'.join(properties) or '无'}）")
        elif value is not None:  # 可选参数显式给 null 视为未提供
            normalized[name] = _check(name, value, spec, notes)  # 归一化 + 校验参数内容
    return normalized, notes


def _check(name: str, value, spec: dict, notes: list[str]):
    """归一化 + 校验单个参数值：能修的修掉并记 notes，不能修的抛 ArgumentError。"""
    kind = spec.get("type")

    if kind in ("integer", "number"):
        value = _as_number(name, value, kind)
        minimum, maximum = spec.get("minimum"), spec.get("maximum")
        if minimum is not None and value < minimum:  # 数量越界不改变意图，夹紧即可
            notes.append(f"参数 {name} 低于下限 {minimum}，已按下限处理")
            value = minimum
        elif maximum is not None and value > maximum:
            notes.append(f"参数 {name} 超过上限 {maximum}，已按上限处理")
            value = maximum

    elif kind == "string":
        if not isinstance(value, str):
            raise ArgumentError(f"参数 {name} 应为字符串，收到 {type(value).__name__}。")
        if spec.get("minLength") and not value.strip():
            raise ArgumentError(f"参数 {name} 不能为空。")

    elif kind == "array":
        if not isinstance(value, list):
            raise ArgumentError(f"参数 {name} 应为数组，收到 {type(value).__name__}。")
        low, high = spec.get("minItems"), spec.get("maxItems")
        if low and len(value) < low:  # 补不齐（不知道该补哪个），只能回喂
            raise ArgumentError(f"参数 {name} 至少需要 {low} 项，只收到 {len(value)} 项。")
        if high and len(value) > high:  # 顺序即模型给的优先级，取前 N 项
            notes.append(f"参数 {name} 超过上限 {high} 项，已取前 {high} 项")
            value = value[:high]

    elif kind == "boolean" and not isinstance(value, bool):
        raise ArgumentError(f"参数 {name} 应为布尔值，收到 {type(value).__name__}。")

    allowed = spec.get("enum")
    if allowed and value not in allowed:
        raise ArgumentError(
            f"参数 {name} 的取值 {value!r} 不在允许范围："
            f"{'、'.join(str(item) for item in allowed)}。"
        )
    return value


def _as_number(name: str, value, kind: str):
    """数字归一化：字符串数字直接转（无损）；其余非数字一律拒绝，小数不取整。"""
    if isinstance(value, str):
        try:
            return int(value) if kind == "integer" else float(value)
        except ValueError:
            raise ArgumentError(f"参数 {name} 的值 {value!r} 不是有效数字。") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArgumentError(f"参数 {name} 应为数字，收到 {type(value).__name__}。")
    if kind == "integer" and not isinstance(value, int):
        raise ArgumentError(f"参数 {name} 应为整数，收到小数 {value}。")
    return value
