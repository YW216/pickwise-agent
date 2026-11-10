"""Langfuse 封装的唯一入口：业务代码只需要认识 production_span、update_current_span
和 ProductionTracing。

文件按依赖顺序排列（底层在前），想按概念理解，按下面的顺序跳读：

1. 两个 ContextVar：当前在哪个节点、是否处于生产追踪
2. production_span 与 update_current_span：业务调用方只用这两个
3. Observation 与 ProductionTracing：节点对象与它的管理器
4. activate_observation：把节点接进 SDK 的上下文，唯一需要懂 SDK 的地方
5. filter_metadata 与三个 mask 函数：导出前的脱敏

两条贯穿全局的约束：

- 追踪出任何问题都不能影响业务：SDK 抛错一律吞掉，业务照常执行。
- 关闭追踪时所有调用退化为空操作：调用方不需要判断追踪有没有开。

SDK 的导出钩子可能在其他线程执行，所以判断"是不是生产环境"读的是 Span 属性，
不是协程里的 ContextVar。Web 服务启动后、处理模型请求前完成初始化。
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

# 生产环境来源标识：导出钩子靠它区分生产与离线评测。
PRODUCTION_SOURCE_KEY = "pickwise_source"

# 允许记录的元数据白名单；不在表里的键一律丢弃。
ALLOWED_METADATA_FIELDS = {
    PRODUCTION_SOURCE_KEY,
    "request_id",
    "agent",
    "tool",
    "tool_call_id",
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
    "react_steps",
    "scenarios",
    "finish_reason",
    "finish_reasons",
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
                metadata={PRODUCTION_SOURCE_KEY: "production"},
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
    """只在生产聊天链路中创建子节点，其他调用直接空跑。

    这里只做守卫：没开追踪、没有父节点、父节点没有客户端时，给调用方一个
    空对象。建节点、激活、结束、失败上报都在 ProductionTracing.span 里。
    """
    parent = current_observation.get()
    if not production_scope.get() or parent is None or parent.tracing is None:
        yield Observation()
        return

    with parent.tracing.span(name, **fields) as observation:
        yield observation


def update_current_span(**fields):
    """更新当前生产节点，不记录 Prompt、回答或异常正文。"""
    observation = current_observation.get()
    if production_scope.get() and observation is not None:
        observation.update(**fields)


# ============================================================
# 3. 元数据处理和数据脱敏
# ============================================================

def filter_metadata(data):
    """按白名单过滤元数据：不在表里的键、非基础类型的值都丢掉。"""
    metadata = {}

    for key, value in data.items():
        if key not in ALLOWED_METADATA_FIELDS:
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
        key = f"langfuse.{kind}.metadata.{PRODUCTION_SOURCE_KEY}"

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

                if metadata_field in ALLOWED_METADATA_FIELDS:
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
        self.status = "success"
        self.failure_stage = None
        self.error_type = None
        self.trace_id = None
        self.span_id = None
        if sdk is not None:
            self.trace_id = sdk.trace_id
            self.span_id = sdk.id

    def update(self, **fields):
        """仅更新白名单元数据，不记录异常正文。"""
        if fields.get("status"):
            self.status = fields["status"]
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

            metadata = filter_metadata(fields)
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
            metadata = filter_metadata(fields)
            metadata[PRODUCTION_SOURCE_KEY] = "production"

            # 固化生产标记供后台导出读取，避免它被运行时脱敏器屏蔽。
            with set_production_scope(False):
                sdk = self.client.start_observation(name=name, metadata=metadata)
            observation = Observation(sdk, tracing=self)
            observation.status = metadata.get("status", "success")
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
        """创建并管理一个 Observation 的完整生命周期。

        离开时如果节点处于失败/降级/溢出状态，把父节点一并标成降级——
        每一层都这么做，状态就沿着调用栈冒到根节点。
        """
        parent = current_observation.get()
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
            if parent is not None and observation.status in {
                "failed", "degraded", "overflow",
            }:
                parent.update(
                    status="degraded",
                    failure_stage=observation.failure_stage or name,
                    error_type=observation.error_type,
                )

    async def aclose(self):
        """将尚未上报的追踪数据刷新到 Langfuse。"""
        if self.client is None:
            return

        try:
            # 将同步 flush 操作放到单独线程执行
            await asyncio.to_thread(self.client.flush)
        except Exception:
            pass
