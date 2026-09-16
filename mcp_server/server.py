"""MCP Server —— PickWise 改造中，当前暂停使用。

背景：
原实现通过 Streamable HTTP 暴露的是**客服工具**
（query_order / query_product / query_logistics / apply_refund）。
随 PickWise 选购业务改造，这些客服工具已从 ``app/agent/tools`` 移除
（order.py / logistics.py / refund.py / user_orders.py / product.py 已删除），
因此本文件暂停使用，避免 import 失败。

待 MCP 接入时（见 ``develop_docs/模块/2-工具系统设计.md`` 3.5 的 use_mcp 占位）：
1. 从 ``app.agent.tools.registry`` 导入 ``TOOL_DEFINITIONS`` 与 ``execute_tool``；
2. 用 FastMCP 注册 9 个 PickWise 工具
   （search_catalog / get_user_favorites / search_by_reviews / get_detail /
   compare_products / retrieve_reviews / retrieve_knowledge /
   retrieve_warranty / turn_lookup）；
3. 由 ``app/agent/tools/manager.py`` 的 ``_init_mcp`` 连接本服务。

启动方式（恢复后）：python mcp_server/server.py
默认监听：http://127.0.0.1:9123/mcp
"""

# 说明：此处不再 import 已移除的客服工具，避免模块加载失败。
# 恢复 MCP 时，参考上方步骤 1-3 重写本文件的工具注册部分。
