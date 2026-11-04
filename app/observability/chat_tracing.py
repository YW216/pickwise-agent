"""聊天根 Trace 的结果整理与收尾。"""


def finish_chat_trace(
    trace, *, idempotency_status, execution_status, answer_status,
    persistence_status, response=None, status=None,
    failure_stage=None, error_type=None, replayed=False, answer_length=None,
):
    """整理聊天结果并结束根节点，不参与聊天执行和幂等判断。

    execution_status / answer_status / persistence_status 必填：三者描述同一轮的
    不同层面，拿不到真实结果时由调用方显式写 "not_executed" 或 "failed"，
    而不是让这里按响应码去猜。
    """
    if idempotency_status == "not_provided":
        idempotency_status = "not_used"
    elif idempotency_status == "finished":
        idempotency_status = "execute"

    if status is None:
        if response is not None:
            if response.status_code >= 500:
                status = "error"
            elif response.status_code >= 400:
                status = "rejected"
            else:
                status = "success"
        elif execution_status == "cancelled":
            status = "cancelled"
        elif execution_status == "rejected":
            status = "rejected"
        elif execution_status == "failed":
            status = "error"
        else:
            status = "success"

    # 幂等账本没写成功时，执行本身是成功的、只是记不下来，统一降级表达。
    if response is not None and response.headers.get("X-Idempotency-Persisted") == "false":
        idempotency_status = "persistence_failed"
        if execution_status == "success":
            execution_status = "degraded"

    if failure_stage is None:
        failure_stage = getattr(trace, "failure_stage", None)

    fields = {
        "status": status,
        "execution_status": execution_status,
        "answer_status": answer_status,
        "persistence_status": persistence_status,
        "idempotency_status": idempotency_status,
        "replayed": replayed,
    }
    if failure_stage:
        fields["failure_stage"] = failure_stage
    if error_type:
        fields["error_type"] = error_type
    if answer_length is not None:
        fields["answer_length"] = answer_length

    trace_id = getattr(trace, "trace_id", None)
    if response is not None and trace_id:
        response.headers["X-Trace-Id"] = trace_id

    trace.update(**fields)
    trace.end()
