"""待确认动作仓储：认领互斥、过期回收、跨用户隔离。

这些行为决定了重复确认会不会重复写库，所以每条都要有断言。
"""

import asyncio
from uuid import uuid4

from app.config.settings import settings
from app.db.action_repo import (
    claim_action,
    create_action,
    finish_action,
    cancel_action,
    get_action,
    init_action_schema,
    list_pending_actions,
    reclaim_stale_actions,
)
from app.db.connection import close_all_pools, connect

USER = f"action-test-{uuid4().hex}"
SESSION = str(uuid4())
ARGS = {"product_ref": "ref-abc"}
PREVIEW = {"name": "戴尔 XPS 13", "price": 7999}


def _cleanup() -> None:
    with connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM pending_actions WHERE user_id = %s", (USER,))
        conn.commit()


def _run(scenario) -> None:
    """执行用例协程；结束时清数据并关池——必须和用例在同一个事件循环里。"""
    async def main():
        try:
            await scenario
        finally:
            _cleanup()
            await close_all_pools()

    asyncio.run(main())


def _move_executing_back(*, action_id: str, seconds: int) -> None:
    """把认领时间推到过去，模拟执行中的进程被杀。"""
    with connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE pending_actions"
                " SET executing_at = now() - make_interval(secs => %s)"
                " WHERE action_id = %s",
                (seconds, action_id),
            )
        conn.commit()


def _expire_now(*, action_id: str) -> None:
    with connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE pending_actions SET expires_at = now() - interval '1 second'"
                " WHERE action_id = %s",
                (action_id,),
            )
        conn.commit()


def test_pending_action_lifecycle_and_repeat_confirm():
    async def scenario():
        init_action_schema(settings.database_url)

        created = await create_action(
            settings.database_url, user_id=USER, session_id=SESSION,
            tool="add_to_favorites", arguments=ARGS, preview=PREVIEW,
        )
        action_id = created["action_id"]

        claimed = await claim_action(
            settings.database_url, action_id=action_id, user_id=USER,
        )
        assert claimed["tool"] == "add_to_favorites"
        assert claimed["arguments"] == ARGS

        # 第二次认领抢不到：这正是"用户连点两次确认"的场景。
        again = await claim_action(
            settings.database_url, action_id=action_id, user_id=USER,
        )
        assert again is None

        await finish_action(
            settings.database_url, action_id=action_id, status="done",
            result={"product_id": "LP-01"},
        )

        record = await get_action(
            settings.database_url, action_id=action_id, user_id=USER,
        )
        assert record["status"] == "done"
        assert record["result"] == {"product_id": "LP-01"}

    _run(scenario())


def test_concurrent_claim_only_one_wins():
    async def scenario():
        init_action_schema(settings.database_url)
        created = await create_action(
            settings.database_url, user_id=USER, session_id=SESSION,
            tool="add_to_favorites", arguments=ARGS, preview=PREVIEW,
        )
        action_id = created["action_id"]

        results = await asyncio.gather(*(
            claim_action(settings.database_url, action_id=action_id, user_id=USER)
            for _ in range(5)
        ))
        winners = [item for item in results if item is not None]
        assert len(winners) == 1

    _run(scenario())


def test_other_user_cannot_claim():
    async def scenario():
        init_action_schema(settings.database_url)
        created = await create_action(
            settings.database_url, user_id=USER, session_id=SESSION,
            tool="add_to_favorites", arguments=ARGS, preview=PREVIEW,
        )
        stranger = f"action-other-{uuid4().hex}"

        stolen = await claim_action(
            settings.database_url, action_id=created["action_id"], user_id=stranger,
        )
        assert stolen is None

        record = await get_action(
            settings.database_url, action_id=created["action_id"], user_id=stranger,
        )
        assert record is None

    _run(scenario())


def test_expired_action_cannot_be_claimed():
    async def scenario():
        init_action_schema(settings.database_url)
        created = await create_action(
            settings.database_url, user_id=USER, session_id=SESSION,
            tool="add_to_favorites", arguments=ARGS, preview=PREVIEW,
        )
        action_id = created["action_id"]
        _expire_now(action_id=action_id)

        assert await claim_action(
            settings.database_url, action_id=action_id, user_id=USER,
        ) is None

        reclaimed = await reclaim_stale_actions(settings.database_url)
        assert reclaimed["expired"] >= 1

        record = await get_action(
            settings.database_url, action_id=action_id, user_id=USER,
        )
        assert record["status"] == "expired"

        listed = await list_pending_actions(settings.database_url, user_id=USER)
        assert action_id not in {item["action_id"] for item in listed}

    _run(scenario())


def test_stuck_executing_is_reclaimed_and_retryable():
    async def scenario():
        init_action_schema(settings.database_url)
        created = await create_action(
            settings.database_url, user_id=USER, session_id=SESSION,
            tool="add_to_favorites", arguments=ARGS, preview=PREVIEW,
        )
        action_id = created["action_id"]

        assert await claim_action(
            settings.database_url, action_id=action_id, user_id=USER,
        ) is not None
        _move_executing_back(action_id=action_id, seconds=600)

        reclaimed = await reclaim_stale_actions(settings.database_url)
        assert reclaimed["failed"] >= 1

        # failed 允许重试，重试的安全性由下游幂等（收藏主键）保证。
        retried = await claim_action(
            settings.database_url, action_id=action_id, user_id=USER,
        )
        assert retried is not None

    _run(scenario())


def test_cancel_then_cannot_confirm():
    async def scenario():
        init_action_schema(settings.database_url)
        created = await create_action(
            settings.database_url, user_id=USER, session_id=SESSION,
            tool="add_to_favorites", arguments=ARGS, preview=PREVIEW,
        )
        action_id = created["action_id"]

        assert await cancel_action(
            settings.database_url, action_id=action_id, user_id=USER,
        ) == "cancelled"

        assert await claim_action(
            settings.database_url, action_id=action_id, user_id=USER,
        ) is None

    _run(scenario())


def test_list_returns_only_actionable():
    async def scenario():
        init_action_schema(settings.database_url)
        first = await create_action(
            settings.database_url, user_id=USER, session_id=SESSION,
            tool="add_to_favorites", arguments=ARGS, preview=PREVIEW,
        )
        second = await create_action(
            settings.database_url, user_id=USER, session_id=SESSION,
            tool="add_to_favorites", arguments=ARGS, preview=PREVIEW,
        )
        await cancel_action(
            settings.database_url, action_id=second["action_id"], user_id=USER,
        )

        listed = await list_pending_actions(settings.database_url, user_id=USER)
        ids = {item["action_id"] for item in listed}
        assert first["action_id"] in ids
        assert second["action_id"] not in ids

    _run(scenario())
