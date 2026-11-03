"""聊天路由：普通响应、SSE、幂等处理与根 Trace。"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.api.auth import AuthenticatedUser
from app.api.deps import get_session_manager, identity_for, require_csrf
from app.api.envelope import error, new_request_id, session_error, success
from app.api.public_text import public_assistant_text
from app.api.schemas import ChatRequest
from app.config.settings import settings
from app.identity import SessionIdentity
from app.observability.chat_tracing import finish_chat_trace
from app.services.idempotency import IdempotencyStore, hash_request, validate_idempotency_key
from app.services.product_refs import resolve_product_context
from app.services.session_manager import ChatOutcome, SessionManager, SessionManagerError


chat_router = APIRouter(prefix="/api/chat", tags=["chat"])


@chat_router.post("")
async def chat(
    body: ChatRequest,
    request: Request,
    user: AuthenticatedUser = Depends(require_csrf),
    manager: SessionManager = Depends(get_session_manager),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    """聊天：支持 Idempotency-Key，防止客户端重试重复执行 Agent。"""
    return await _handle_chat(body, request, user, manager, idempotency_key)


@chat_router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    user: AuthenticatedUser = Depends(require_csrf),
    manager: SessionManager = Depends(get_session_manager),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    """过程实时 SSE；答案生成并保存后分段发送，不是模型 Token 流式。"""
    return await _handle_chat(body, request, user, manager, idempotency_key, stream=True)


async def _handle_chat(body, request, user, manager, idempotency_key, *, stream=False):
    """按检查、幂等判断、执行和响应的顺序编排聊天请求。"""
    request_id = new_request_id()
    mode = "stream" if stream else "sync"
    trace = request.app.state.tracing.start(
        "pickwise.chat",
        request_id=request_id,
        mode=mode,
        message_length=len(body.message),
    )
    if idempotency_key is not None:
        idempotency_status = "provided"
    else:
        idempotency_status = "not_provided"

    try:
        identity = identity_for(user, body.session_id)
    except HTTPException:
        finish_chat_trace(
            trace,
            idempotency_status=idempotency_status,
            status="rejected",
            failure_stage="identity",
            error_type="HTTPException",
            execution_status="not_started",
        )
        raise

    with trace.activate(user_id=identity.user_id, session_id=identity.session_id):
        product_context = []
        for ref in body.product_refs:
            item = resolve_product_context(identity.user_id, identity.session_id, ref)
            if item is None:
                response = error(
                    422, "PRODUCT_REF_INVALID",
                    "推荐商品已失效，请重新发起推荐", request_id,
                )
                finish_chat_trace(
                    trace,
                    idempotency_status=idempotency_status,
                    response=response,
                    failure_stage="product_ref",
                    error_type="PRODUCT_REF_INVALID",
                )
                return response
            product_context.append(item)

        decision = await _check_chat_idempotency(
            request.app.state.idempotency_store,
            identity, body, idempotency_key, request_id, stream=stream,
        )

    if decision.response is not None:
        replayed = decision.status == "replay"
        failure_stage = None if replayed else "idempotency"
        finish_chat_trace(
            trace,
            idempotency_status=decision.status,
            response=decision.response,
            failure_stage=failure_stage,
            error_type=decision.error_type,
            replayed=replayed,
        )
        return decision.response

    events = asyncio.Queue(maxsize=256) if stream else None
    execution = None
    if stream:
        try:
            # 在根节点激活期间创建执行任务，使它继承同一追踪上下文。
            with trace.activate(user_id=identity.user_id, session_id=identity.session_id):
                execution = await manager.chat_stream(
                    identity, body.message, events, product_context=product_context,
                )
        except Exception as exc:
            if isinstance(exc, SessionManagerError):
                response = session_error(exc, request_id)
            else:
                response = error(503, "CHAT_UNAVAILABLE", "会话服务暂时不可用", request_id)
            response = await _save_idempotency_result(
                request, identity, decision.key, decision.request_hash, response,
            )
            idempotency_status = "finished" if decision.key else decision.status
            finish_chat_trace(
                trace,
                idempotency_status=idempotency_status,
                response=response,
                failure_stage="session",
                error_type=type(exc).__name__,
            )
            return response

    async def run():
        answer_length = None
        persistence_status = None
        try:
            with trace.activate(user_id=identity.user_id, session_id=identity.session_id):
                try:
                    if execution is not None:
                        outcome = await execution
                    else:
                        outcome = await manager.chat(
                            identity, body.message, product_context=product_context,
                        )
                except SessionManagerError as exc:
                    response = session_error(exc, request_id)
                    failure_stage = "session"
                    error_type = type(exc).__name__
                except Exception as exc:
                    response = error(
                        500, "CHAT_INTERNAL_ERROR", "服务内部错误，请稍后重试", request_id,
                    )
                    failure_stage = "chat"
                    error_type = type(exc).__name__
                else:
                    response = _chat_response(identity, outcome, request_id)
                    failure_stage = "agent" if outcome.failed else None
                    error_type = "CHAT_FAILED" if outcome.failed else None
                    answer_length = len(outcome.reply)
                    persistence_status = "success" if outcome.persisted else "failed"

                response = await _save_idempotency_result(
                    request, identity, decision.key, decision.request_hash, response,
                )
                if stream:
                    # 执行和幂等记账结束后，通知 SSE 生成器发送最终答案。
                    while events.full():
                        events.get_nowait()
                    events.put_nowait(None)

                idempotency_status = "finished" if decision.key else decision.status
                finish_chat_trace(
                    trace,
                    idempotency_status=idempotency_status,
                    response=response,
                    failure_stage=failure_stage,
                    error_type=error_type,
                    answer_length=answer_length,
                    persistence_status=persistence_status,
                )
                return response
        except BaseException as exc:
            finish_chat_trace(
                trace,
                idempotency_status=decision.status,
                failure_stage="chat",
                error_type=type(exc).__name__,
            )
            raise

    if not stream:
        return await run()

    # 浏览器断连只停止传输，后台任务继续完成生成、保存和幂等记账。
    with trace.activate(user_id=identity.user_id, session_id=identity.session_id):
        task = asyncio.create_task(run())
    request.app.state.stream_tasks.add(task)
    task.add_done_callback(request.app.state.stream_tasks.discard)
    return StreamingResponse(
        _stream_events(events, task), media_type="text/event-stream", headers=_SSE_HEADERS,
    )


@dataclass
class _ChatIdempotencyResult:
    """幂等检查的返回数据；response 有值时，入口直接返回而不执行 Agent。"""

    key: str | None
    request_hash: str | None = None
    status: str = "not_provided"
    response: Response | None = None
    error_type: str | None = None


async def _check_chat_idempotency(
    store: IdempotencyStore,
    identity: SessionIdentity,
    body: ChatRequest,
    idempotency_key: str | None,
    request_id: str,
    *,
    stream: bool,
) -> _ChatIdempotencyResult:
    """校验并抢占幂等请求，统一返回继续执行、拒绝或重放的判定。"""
    result = _ChatIdempotencyResult(key=idempotency_key)
    if idempotency_key is None:
        return result

    try:
        result.key = validate_idempotency_key(idempotency_key)
    except ValueError as exc:
        result.status = "invalid"
        result.error_type = "IDEMPOTENCY_KEY_INVALID"
        result.response = error(422, result.error_type, str(exc), request_id)
        return result

    result.request_hash = hash_request({
        "session_id": identity.session_id,
        "message": body.message,
        "product_refs": body.product_refs,
    })
    try:
        claim = await store.begin(
            identity.user_id, identity.session_id, result.key, result.request_hash,
            ttl_seconds=settings.idempotency_ttl_seconds,
        )
    except Exception:
        result.status = "unavailable"
        result.error_type = "IdempotencyStoreError"
        result.response = error(
            503, "IDEMPOTENCY_UNAVAILABLE", "请求去重服务暂时不可用", request_id,
        )
        return result

    result.status = claim.action
    if claim.action == "execute":
        return result
    if claim.action == "conflict":
        result.error_type = "IDEMPOTENCY_KEY_REUSED"
        result.response = error(
            409, result.error_type, "同一个 Idempotency-Key 不能用于不同请求", request_id,
        )
        return result
    if claim.action == "in_progress":
        result.error_type = "REQUEST_IN_PROGRESS"
        result.response = error(
            409, result.error_type, "相同请求正在处理中，请稍后重试", request_id,
        )
        result.response.headers["Retry-After"] = "1"
        return result

    response = JSONResponse(
        status_code=claim.status_code or 200,
        content=claim.response or {},
    )
    response.headers["X-Idempotent-Replay"] = "true"
    if response.status_code >= 400:
        result.error_type = "REPLAYED_FAILURE"
    if stream and response.status_code < 400:
        response = StreamingResponse(
            _answer_events(claim.response, replayed=True),
            media_type="text/event-stream", headers=_SSE_HEADERS,
        )
    result.response = response
    return result


async def _save_idempotency_result(request, identity, idempotency_key, request_hash, response):
    """保存本次响应供重试重放；记账失败时保留已经生成的业务结果。"""
    if idempotency_key is not None and request_hash is not None:
        try:
            payload = json.loads(response.body.decode("utf-8"))
            await request.app.state.idempotency_store.finish(
                identity.user_id,
                identity.session_id,
                idempotency_key,
                request_hash,
                status="succeeded" if response.status_code < 400 else "failed",
                status_code=response.status_code,
                response=payload,
            )
        except Exception:
            # Agent 已经执行完成，不能因为幂等账本写失败而再次执行；
            # 对客户端保留本次结果，并明确标注账本未写入。
            response.headers["X-Idempotency-Persisted"] = "false"
    return response


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


_HEARTBEAT_SECONDS = 15.0


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _answer_events(payload, *, replayed=False):
    data = payload.get("data") or {}
    reply = public_assistant_text(data.get("reply", ""))
    data = {**data, "reply": reply}
    # 保留标点和换行；超长段落继续按 80 字切分，拼接必须与原文完全一致。
    chunks = re.findall(r"[^。！？\n]{1,80}[。！？\n]?|[。！？\n]", reply)
    for seq, text in enumerate(chunks):
        yield _sse("answer.delta", {"text": text, "seq": seq})
        await asyncio.sleep(0)
    if payload.get("success") is not True:
        failure = payload.get("error") or {}
        yield _sse("error", {**failure, "request_id": payload.get("request_id")})
    yield _sse("done", {**data, "request_id": payload.get("request_id"),
                        "failed": payload.get("success") is not True,
                        "replayed": replayed})


async def _stream_events(events, task):
    pending = asyncio.create_task(events.get())
    try:
        while True:
            ready, _ = await asyncio.wait(
                {pending, task}, timeout=_HEARTBEAT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not ready:
                yield ": ping\n\n"
            elif pending in ready:
                event = pending.result()
                if event is None:
                    break
                yield _sse(event["event"], event["data"])
                pending = asyncio.create_task(events.get())
        response = task.result()
        payload = json.loads(response.body)
        if response.headers.get("X-Idempotency-Persisted") == "false":
            payload.setdefault("data", {}).setdefault("warnings", []).append("IDEMPOTENCY_PERSISTENCE_FAILED")
        async for frame in _answer_events(payload):
            yield frame
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


def _chat_response(identity: SessionIdentity, outcome: ChatOutcome,
                   request_id: str):
    """ChatOutcome → HTTP（§4.5）：

    - 影响答案的失败 → 503，data 仍带已生成的兜底 reply；
    - 仅附属失败（STM/落盘）→ 200，warnings 给稳定 code（不伪装成功保存）。
    """
    data = {
        "session_id": identity.session_id,
        "reply": public_assistant_text(outcome.reply),
        "warnings": list(outcome.warnings),
    }
    if outcome.failed:
        return error(503, "CHAT_FAILED",
                     "本轮回答生成失败，请稍后重试", request_id, data)
    if not outcome.persisted:
        data["warnings"] = [*data["warnings"], "PERSISTENCE_FAILED"]
    return success(data, request_id)
