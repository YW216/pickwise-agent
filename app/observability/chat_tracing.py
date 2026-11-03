"""聊天根 Trace 的结果整理与收尾。"""


def finish_chat_trace(
    trace, *, idempotency_status, response=None, status=None,
    execution_status=None, answer_status=None, persistence_status=None,
    failure_stage=None, error_type=None, replayed=False, **extra_fields,
):
    """整理聊天结果并结束根节点，不参与聊天执行和幂等判断。"""
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

    if execution_status is None:
        if replayed and response is not None and response.status_code < 400:
            execution_status = "success"
        elif response is not None and 400 <= response.status_code < 500:
            execution_status = "rejected"
        elif response is not None and response.status_code >= 500:
            execution_status = "failed"
        elif getattr(trace, "degraded", False):
            execution_status = "degraded"
        else:
            execution_status = "success"

    if getattr(trace, "degraded", False) and execution_status == "success":
        execution_status = "degraded"

    if answer_status is None:
        if response is not None and response.status_code < 400:
            answer_status = "success"
        elif execution_status in {"rejected", "cancelled"}:
            answer_status = "not_generated"
        else:
            answer_status = "failed"

    if persistence_status is None:
        if response is not None and response.status_code < 400 and not replayed:
            persistence_status = "success"
        else:
            persistence_status = "not_executed"

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

    fields.update(extra_fields)

    trace.update(**fields)
    trace.end()
