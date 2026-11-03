
"""简单的 Langfuse 封装，不在这里安装聊天埋点。

只有显式进入生产环境追踪范围时才进行数据脱敏；
离线评测不受影响。

SDK 的导出钩子也可能在其他线程中执行，
因此导出时通过 Span 属性判断环境，而不是依赖 ContextVar。

Web 服务启动后、处理模型请求前，应完成实例初始化。
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar


# ============================================================
# 1. 常量和配置
# ============================================================

# 标记当前是否处于生产环境 Trace 范围
production_scope = ContextVar(
    "pickwise_production_trace",
    default=False,
)

# 当前生产节点随协程传播，供 Router 和专家创建子节点、更新安全元数据。
current_observation = ContextVar("pickwise_current_observation", default=None)

# 生产环境来源标识
SOURCE = "pickwise_source"

# 允许记录的元数据字段白名单
FIELDS = {
    SOURCE,
    "request_id",
    "agent",
    "tool",
    "tool_call_id",
    "purpose",
    "mode",
    "status",
    "error_type",
    "failure_stage",
    "execution_status",
    "answer_status",
    "persistence_status",
    "idempotency_status",
    "message_length",
    "answer_length",
    "result_count",
    "recommendation_count",
    "react_steps",
    "scenarios",
    "degraded",
    "finish_reason",
    "finish_reasons",
    "agent_reused",
    "agent_initialized",
    "state_restored",
    "memory",
    "memory_status",
    "store",
    "cache_hit",
    "replayed",
}

# OpenTelemetry Span 允许保留的属性
OTEL_KEEP_ATTRIBUTES = {
    "langfuse.observation.type",
    "langfuse.observation.level",
    "langfuse.observation.model.name",
    "langfuse.observation.usage_details",
    "langfuse.observation.cost_details",
    "langfuse.environment",
    "langfuse.trace.name",
    "langfuse.trace.tags",
    "user.id",
    "session.id",
}


# ============================================================
# 2. 上下文工具
# ============================================================

@contextmanager
def set_production_scope(enabled):
    """临时切换运行时脱敏范围，退出时恢复原来的协程上下文。"""
    token = production_scope.set(enabled)
    try:
        yield
    finally:
        production_scope.reset(token)


@contextmanager
def activate_observation(observation, *, user_id=None, session_id=None):
    """激活已有追踪，统一管理业务脱敏范围和 SDK 上下文。"""
    with set_production_scope(True):
        if observation.sdk is None:
            yield observation
            return

        contexts = ExitStack()
        try:
            from langfuse import propagate_attributes
            from opentelemetry.trace import use_span

            # SDK 4.15.3 没有 observe()；直接激活底层真实 Span。
            # 禁止自动写入异常正文，也不在退出时结束跨任务复用的根节点。
            contexts.enter_context(use_span(
                observation.sdk._otel_span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            ))
            token = current_observation.set(observation)
            contexts.callback(current_observation.reset, token)
            contexts.enter_context(propagate_attributes(
                user_id=user_id,
                session_id=session_id,
                metadata={SOURCE: "production"},
            ))
        except Exception:
            # SDK 上下文不可用时，业务仍在脱敏范围内执行一次。
            pass

        try:
            yield observation
        finally:
            # 只清理 SDK 上下文，不把业务异常传给 SDK 自动记录。
            try:
                contexts.close()
            except Exception:
                pass


@contextmanager
def production_span(name, **fields):
    """只在生产聊天链路中创建子节点，其他调用直接空跑。"""
    parent = current_observation.get()
    if not production_scope.get() or parent is None or parent.tracing is None:
        observation = Observation()
        yield observation
        return

    observation = None
    try:
        with parent.tracing.span(name, **fields) as observation:
            yield observation
    finally:
        # 子步骤发生降级时逐层通知父节点，最终由聊天根节点给出整体状态。
        if observation is not None and observation.degraded:
            parent.update(
                status="degraded",
                degraded=True,
                failure_stage=observation.failure_stage or name,
                error_type=observation.error_type,
            )


def update_current_span(**fields):
    """更新当前生产节点，不记录 Prompt、回答或异常正文。"""
    observation = current_observation.get()
    if production_scope.get() and observation is not None:
        observation.update(**fields)


# ============================================================
# 3. 元数据处理和数据脱敏
# ============================================================

def get_metadata(data):
    """只保留白名单中的基础类型元数据。"""
    metadata = {}

    for key, value in data.items():
        if key not in FIELDS:
            continue

        if not isinstance(value, (str, int, float, bool)):
            continue

        # 结束原因来自供应商响应，只保留协议中的固定值。
        if key == "finish_reason" and value not in {
            "stop", "length", "tool_calls", "content_filter", "function_call",
        }:
            continue

        metadata[key] = value

    return metadata


def mask_content(*, data, **kwargs):
    """处于生产环境追踪范围时，屏蔽原始内容。"""
    if production_scope.get():
        return "[content omitted]"

    return data


def is_production_span(attributes):
    """只读取 Span 属性；后台导出线程不能依赖协程 ContextVar。"""
    for kind in ("trace", "observation"):
        key = f"langfuse.{kind}.metadata.{SOURCE}"

        if attributes.get(key) == "production":
            return True

    return False


def should_export_span(span):
    """生产环境 Span 如果包含异常详情，则禁止导出。"""
    attributes = span.attributes or {}

    # 非生产环境的 Span 不受此规则影响
    if not is_production_span(attributes):
        return True

    # 不导出包含异常事件或异常描述的 Span
    if span.events:
        return False

    if span.status.description:
        return False

    return True


def mask_otel_spans(params):
    """删除生产环境 OpenTelemetry Span 中不允许导出的属性。"""
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}

    for span_id, span in params.spans.items():
        attributes = span.attributes or {}

        # 离线评测和非生产环境 Span 保持不变
        if not is_production_span(attributes):
            continue

        attributes_to_remove = []

        for key in attributes:
            # 保留明确允许的属性
            if key in OTEL_KEEP_ATTRIBUTES:
                continue

            # 保留 Langfuse 内部属性
            if key.startswith("langfuse.internal."):
                continue

            # 保留白名单中的元数据
            if ".metadata." in key:
                metadata_field = key.rsplit(".metadata.", 1)[-1]

                if metadata_field in FIELDS:
                    continue

            # 其他属性全部删除
            attributes_to_remove.append(key)

        patches[span_id] = OtelSpanPatch(
            delete_attributes=attributes_to_remove
        )

    return MaskOtelSpansResult(span_patches=patches)


# ============================================================
# 4. Observation：SDK 薄包装
# ============================================================

class Observation:
    """隔离 SDK 更新和结束操作的异常，提供关闭追踪时的空对象。"""

    def __init__(self, sdk=None, *, tracing=None):
        self.sdk = sdk
        self.tracing = tracing
        self.finish_reasons = []
        self.degraded = False
        self.failure_stage = None
        self.error_type = None
        self.trace_id = None
        self.span_id = None
        if sdk is not None:
            self.trace_id = sdk.trace_id
            self.span_id = sdk.id

    def update(self, **fields):
        """仅更新白名单元数据，不记录异常正文。"""
        if fields.get("degraded") is True or fields.get("status") in {
            "failed", "degraded", "overflow",
        }:
            self.degraded = True
        if fields.get("failure_stage"):
            self.failure_stage = fields["failure_stage"]
        if fields.get("error_type"):
            self.error_type = fields["error_type"]

        if self.sdk is None:
            return

        try:
            level = "DEFAULT"
            if fields.get("error_type"):
                level = "ERROR"

            metadata = get_metadata(fields)
            # 保存本节点内每次模型调用的固定结束原因，避免 ReAct 后续步骤覆盖前一步。
            if "finish_reason" in metadata:
                self.finish_reasons.append(metadata["finish_reason"])
                metadata["finish_reasons"] = ",".join(self.finish_reasons)

            # SDK 对 metadata 也调用内容脱敏器；这里只写已过滤的元数据。
            with set_production_scope(False):
                self.sdk.update(metadata=metadata, level=level)
        except Exception:
            pass

    def activate(self, *, user_id=None, session_id=None):
        """兼容聊天根节点接口，上下文管理统一交给外部工具。"""
        return activate_observation(self, user_id=user_id, session_id=session_id)

    def end(self):
        """结束追踪；先释放引用，防止重复结束或继续更新。"""
        if self.sdk is None:
            return

        sdk = self.sdk
        self.sdk = None
        try:
            sdk.end()
        except Exception:
            pass


# ============================================================
# 5. ProductionTracing：管理生产环境追踪
# ============================================================

class ProductionTracing:
    """负责初始化 Langfuse，以及创建和管理生产环境 Observation。"""

    def __init__(
        self,
        *,
        enabled,
        public_key,
        secret_key,
        base_url,
        environment,
    ):
        self.public_key = public_key
        self.client = None

        # 未开启追踪或缺少密钥时，不初始化客户端
        if not enabled or not public_key or not secret_key:
            return

        try:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=base_url or None,
                environment=environment,
                mask=mask_content,
                mask_otel_spans=mask_otel_spans,
                should_export_span=should_export_span,
            )

        except Exception:
            # Langfuse 初始化失败不能阻止应用启动
            self.client = None

    @property
    def enabled(self):
        """判断 Langfuse 客户端是否初始化成功。"""
        return self.client is not None

    def start(self, name, **fields):
        """创建根节点或子节点；生命周期由调用方或 span() 管理。"""
        if self.client is None:
            return Observation()

        try:
            metadata = get_metadata(fields)
            metadata[SOURCE] = "production"

            # 固化生产标记供后台导出读取，避免它被运行时脱敏器屏蔽。
            with set_production_scope(False):
                sdk = self.client.start_observation(name=name, metadata=metadata)
            observation = Observation(sdk, tracing=self)
            observation.degraded = (
                metadata.get("degraded") is True
                or metadata.get("status") in {"failed", "degraded", "overflow"}
            )
            observation.failure_stage = metadata.get("failure_stage")
            observation.error_type = metadata.get("error_type")
            return observation
        except Exception:
            return Observation()

    @contextmanager
    def span(
        self,
        name,
        *,
        user_id=None,
        session_id=None,
        **fields,
    ):
        """创建并管理一个 Observation 的完整生命周期。"""
        # 创建 Observation
        observation = self.start(name, **fields)

        try:
            # 激活 Observation，执行其中的业务代码
            with activate_observation(
                observation,
                user_id=user_id,
                session_id=session_id,
            ):
                yield observation

        except BaseException as exc:
            # 只记录异常类型，不记录异常正文
            observation.update(
                error_type=type(exc).__name__,
                status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
            )
            raise

        finally:
            # 无论成功还是失败，都结束 Observation
            observation.end()

    async def aclose(self):
        """将尚未上报的追踪数据刷新到 Langfuse。"""
        if self.client is None:
            return

        try:
            # 将同步 flush 操作放到单独线程执行
            await asyncio.to_thread(self.client.flush)
        except Exception:
            pass
