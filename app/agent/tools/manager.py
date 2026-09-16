"""ToolManager：管理本地工具（PickWise）。

当前范围：**只实现本地工具**（mock 数据）。
MCP 后续接入时在此扩展 `_init_mcp`；`use_mcp` 参数保留占位，
保证将来加 MCP 不破坏调用方。

职责：
- 加载本地工具定义（registry.TOOL_DEFINITIONS）
- 按白名单过滤工具（子 Agent 工具隔离）
- 提供统一的 tool_definitions（给模型）与 execute_tool（执行）
"""

from typing import Optional

from app.agent.tools.registry import TOOL_DEFINITIONS as LOCAL_TOOL_DEFINITIONS
from app.agent.tools.registry import execute_tool as local_execute_tool
from app.agent.tools.registry import to_tool_message_content


class ToolManager:
    """聚合本地工具，提供统一的工具定义与调度接口。"""

    def __init__(
        self,
        allowed_tools: Optional[set] = None,
        use_mcp: bool = False,
        mcp_server_url: str = "",
    ):
        """初始化工具管理器。

        Args:
            allowed_tools：工具白名单（用于子 Agent 隔离），None 表示不过滤
            use_mcp：是否启用 MCP（当前恒 False，占位供后续扩展）
            mcp_server_url：MCP Server 地址（后续用）
        """
        self._tool_defs: list[dict] = []
        self._tool_source: dict[str, str] = {}

        # MCP 暂不实现：use_mcp=True 时降级为本地工具并提示
        if use_mcp and mcp_server_url:
            print("⚠️  [MCP] 当前版本未实现 MCP，已降级使用本地工具")
        self._init_local()

        if allowed_tools is not None:
            self._filter_tools(allowed_tools)

    def _init_local(self):
        """只加载本地工具。"""
        self._tool_defs = list(LOCAL_TOOL_DEFINITIONS)
        for td in self._tool_defs:
            self._tool_source[td["function"]["name"]] = "local"

    def _filter_tools(self, allowed: set):
        """只保留白名单中的工具，用于子 Agent 工具隔离。"""
        self._tool_defs = [
            d for d in self._tool_defs if d["function"]["name"] in allowed
        ]
        self._tool_source = {
            k: v for k, v in self._tool_source.items() if k in allowed
        }

    @property
    def tool_definitions(self) -> list[dict]:
        """给模型的工具定义（OpenAI function calling 格式）。"""
        return self._tool_defs

    @property
    def tool_names(self) -> list[str]:
        """当前可用工具名列表。"""
        return [d["function"]["name"] for d in self._tool_defs]

    def execute_tool(self, name: str, arguments: dict) -> dict:
        """执行工具，返回统一信封 dict。"""
        source = self._tool_source.get(name)

        if source == "local":
            return local_execute_tool(name, arguments)

        return {"success": False, "error": f"未知工具: {name}", "data": {}}

    def execute_tool_as_message(self, name: str, arguments: dict) -> str:
        """执行工具并把结果序列化为 tool 消息的 content 字符串。"""
        return to_tool_message_content(self.execute_tool(name, arguments))

    def close(self):
        """清理资源（MCP 接入后用于关闭连接）。"""
        return None
