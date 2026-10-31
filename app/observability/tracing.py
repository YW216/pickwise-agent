
"""简单的 Langfuse 封装，不在这里安装聊天埋点。

只有显式进入生产环境追踪范围时才进行数据脱敏；
离线评测不受影响。

SDK 的导出钩子也可能在其他线程中执行，
因此导出时通过 Span 属性判断环境，而不是依赖 ContextVar。

Web 服务启动后、处理模型请求前，应完成实例初始化。
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar


# ============================================================
# 1. 常量和配置
# ============================================================

# 标记当前是否处于生产环境 Trace 范围
production_scope = ContextVar(
    "pickwise_production_trace",
    default=False,
)

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
# 2. 元数据处理和数据脱敏
# ============================================================

def get_metadata(data):
    """只保留白名单中的基础类型元数据。"""
    metadata = {}

    for key, value in data.items():
        if key not in FIELDS:
            continue

        if not isinstance(value, (str, int, float, bool)):
            continue

        metadata[key] = value

    return metadata


def mask_content(*, data, **kwargs):
    """处于生产环境追踪范围时，屏蔽原始内容。"""
    if production_scope.get():
        return "[content omitted]"

    return data


def is_production_span(attributes):
    """判断当前 OpenTelemetry Span 是否属于生产环境。"""
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
# 3. Observation：管理单个追踪对象
# ============================================================

class Observation:
    """封装 Langfuse Observation 的创建、激活、更新和结束。"""

    def __init__(self, sdk=None):
        self.sdk = sdk

        if sdk is None:
            self.trace_id = None
            self.span_id = None
        else:
            self.trace_id = sdk.trace_id
            self.span_id = sdk.id

    def update(self, **fields):
        """更新 Observation 的元数据和状态。"""
        if self.sdk is None:
            return

        # 更新元数据时暂时关闭生产环境内容屏蔽标记
        token = production_scope.set(False)

        try:
            if fields.get("error_type"):
                level = "ERROR"
            else:
                level = "DEFAULT"

            self.sdk.update(
                metadata=get_metadata(fields),
                level=level,
            )

        except Exception:
            # 追踪失败不能影响业务执行
            pass

        finally:
            production_scope.reset(token)

    @contextmanager
    def activate(self, *, user_id=None, session_id=None):
        """在当前执行上下文中激活 Observation。"""
        if self.sdk is None:
            yield self
            return

        # 标记当前执行过程属于生产环境
        token = production_scope.set(True)

        try:
            # 初始化 Langfuse 和 OpenTelemetry 上下文
            from langfuse import propagate_attributes
            from opentelemetry.trace import (
                NonRecordingSpan,
                SpanContext,
                TraceFlags,
                use_span,
            )

            # 根据当前 Observation 的 ID 构造父 Span
            parent = NonRecordingSpan(
                SpanContext(
                    int(self.trace_id, 16),
                    int(self.span_id, 16),
                    is_remote=False,
                    trace_flags=TraceFlags(TraceFlags.SAMPLED),
                )
            )

        except Exception:
            # 上下文初始化失败时，仍然允许业务代码正常执行
            try:
                yield self
            finally:
                production_scope.reset(token)

            return

        try:
            # 激活父 Span，避免自动记录异常文本和堆栈
            with use_span(
                parent,
                record_exception=False,
                set_status_on_exception=False,
            ):
                # 传递用户、会话和生产环境元数据
                with propagate_attributes(
                    user_id=user_id,
                    session_id=session_id,
                    metadata={SOURCE: "production"},
                ):
                    yield self

        finally:
            # 恢复进入上下文之前的状态
            production_scope.reset(token)

    def end(self):
        """结束当前 Observation。"""
        if self.sdk is None:
            return

        sdk = self.sdk
        self.sdk = None

        try:
            sdk.end()
        except Exception:
            # 结束追踪失败不影响业务
            pass


# ============================================================
# 4. ProductionTracing：管理生产环境追踪
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
        if not enabled:
            return

        if not public_key or not secret_key:
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
        """创建一个生产环境 Observation。"""
        if self.client is None:
            return Observation()

        # 创建 Observation 时只传递白名单元数据
        token = production_scope.set(False)

        try:
            metadata = get_metadata(fields)
            metadata[SOURCE] = "production"

            sdk = self.client.start_observation(
                name=name,
                metadata=metadata,
            )

            return Observation(sdk)

        except Exception:
            # 创建追踪失败时返回空 Observation
            return Observation()

        finally:
            production_scope.reset(token)

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
            with observation.activate(
                user_id=user_id,
                session_id=session_id,
            ):
                yield observation

        except BaseException as exc:
            # 只记录异常类型，不记录异常正文
            observation.update(
                error_type=type(exc).__name__
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
