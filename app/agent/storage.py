import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

SESSION_VERSION = 4


def _valid_state(messages, summary, products) -> bool:
    if not isinstance(messages, list) or any(
        not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "tool"}
        for message in messages
    ):
        return False
    if messages and messages[0].get("role") != "user":
        # 窗口起点必须是 user turn：压缩只在 user turn 边界切分（C2），
        # 会话第一条消息恒为 user（chat() 先追加 user 再执行 Agent）。
        return False
    if summary is not None and (not isinstance(summary, str) or not summary.strip()):
        return False
    if not isinstance(products, dict) or any(
        not isinstance(key, str) or not isinstance(value, dict)
        for key, value in products.items()
    ):
        return False
    return True


def save_session(
    path: str,
    messages: list[dict],
    summary: Optional[str],
    short_term_memory: Optional[dict] = None,
    products: Optional[dict] = None,
) -> None:
    """把对话状态原子写入 JSON 文件。

    messages 是工作历史（未压缩部分；已压缩段以 summary 承载）；
    products 是商品记忆（mentioned-products，跨压缩累积）；
    short_term_memory 为短期记忆的序列化数据（第7期）。
    """
    if not _valid_state(messages, summary, products or {}):
        raise ValueError("会话 messages/summary/products 不合法")
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": SESSION_VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "messages": messages,
        "products": products or {},
        "short_term_memory": short_term_memory,
    }

    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, file_path)


def load_session(path: str) -> Optional[dict]:
    """读取会话文件。不存在或损坏都返回 None（降级为新会话）。"""
    file_path = Path(path)
    if not file_path.exists():
        return None

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️  会话文件损坏，已忽略（{e}）")
        return None

    if not isinstance(data, dict) or data.get("version") != SESSION_VERSION or "messages" not in data:
        print("⚠️  会话文件格式不识别，已忽略")
        return None

    messages = data.get("messages", [])
    products = data.get("products", {})
    if not _valid_state(messages, data.get("summary"), products):
        print("⚠️  会话 messages/summary/products 不合法，已忽略")
        return None

    return {
        "summary": data.get("summary"),
        "messages": messages,
        "products": products,
        "short_term_memory": data.get("short_term_memory"),
    }


def delete_session(path: str) -> None:
    """删除会话文件，不存在时静默。

    删除属清理性质：受限环境（删除被拦截）下降级为改名移开——
    保证旧会话不再被加载，且不因清理失败中断主流程。
    """
    file_path = Path(path)
    if not file_path.exists():
        return
    try:
        file_path.unlink()
    except OSError:
        try:
            file_path.replace(Path(str(file_path) + ".bak"))
        except OSError as e:
            print(f"⚠️  会话文件清理失败（不影响运行）: {e}")
