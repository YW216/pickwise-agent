"""聊天根 Trace 的结果整理与收尾。"""


def finish_chat_trace(
    trace, *, idempotency_status, response=None, status=None,
    failure_stage=None, error_type=None, replayed=False, **extra_fields,
):
    """整理聊天结果并结束根节点，不参与聊天执行和幂等判断。"""
    if status is None:
        if response is None or response.status_code >= 500:
            status = "error"
        elif response.status_code >= 400:
            status = "rejected"
        else:
            status = "success"

    fields = {
        "status": status,
        "replayed": replayed,
        "idempotency_status": idempotency_status,
    }
    if failure_stage:
        fields["failure_stage"] = failure_stage
    if error_type:
        fields["error_type"] = error_type

    if response is not None:
        if replayed:
            fields["execution_status"] = "skipped"
            if response.status_code < 400:
                fields["answer_status"] = "success"
        elif response.status_code < 400:
            fields["execution_status"] = "success"
            fields["answer_status"] = "success"
        elif failure_stage in {"identity", "product_ref", "idempotency", "session"}:
            fields["execution_status"] = "not_started"
        elif response.status_code >= 500:
            fields["execution_status"] = "failed"
            if failure_stage in {"agent", "chat"}:
                fields["answer_status"] = "failed"
        else:
            fields["execution_status"] = "rejected"

        if response.headers.get("X-Idempotency-Persisted") == "false":
            fields["idempotency_status"] = "persistence_failed"

    fields.update(extra_fields)

    trace.update(**fields)
    trace.end()
