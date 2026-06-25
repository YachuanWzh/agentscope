# -*- coding: utf-8 -*-
"""Tests for the in-memory checkpointer."""
import time
from unittest import IsolatedAsyncioTestCase

from agentscope.checkpoint import (
    Checkpoint,
    CheckpointerBase,
    MemoryCheckpointer,
    RetentionPolicy,
)
from agentscope.state import AgentState
from agentscope.message import UserMsg


def _cp(thread: str, parent: str | None = None, step: int = 0) -> Checkpoint:
    """Build a checkpoint with a fresh state for the given thread."""
    return Checkpoint(
        thread_id=thread,
        parent_checkpoint_id=parent,
        step=step,
        state=AgentState(session_id=thread),
    )


class MemoryCheckpointerTest(IsolatedAsyncioTestCase):
    """The in-memory checkpointer tests."""

    def test_is_checkpointer(self) -> None:
        """MemoryCheckpointer is a CheckpointerBase."""
        self.assertIsInstance(MemoryCheckpointer(), CheckpointerBase)

    async def test_put_get(self) -> None:
        """put stores a checkpoint retrievable by id; returns its id."""
        cpr = MemoryCheckpointer()
        cp = _cp("t1")
        returned = await cpr.put(cp)
        self.assertEqual(returned, cp.checkpoint_id)

        got = await cpr.get("t1", cp.checkpoint_id)
        self.assertIsNotNone(got)
        self.assertEqual(got.checkpoint_id, cp.checkpoint_id)

        self.assertIsNone(await cpr.get("t1", "missing"))
        self.assertIsNone(await cpr.get("other", cp.checkpoint_id))

    async def test_snapshot_isolation(self) -> None:
        """Mutating the original state after put does not leak into store."""
        cpr = MemoryCheckpointer()
        state = AgentState(session_id="t1")
        state.context.append(UserMsg(name="user", content="hi"))
        cp = Checkpoint(thread_id="t1", state=state)
        await cpr.put(cp)

        # Mutate the original objects after the put.
        state.context.append(UserMsg(name="user", content="leak?"))

        stored = await cpr.get("t1", cp.checkpoint_id)
        self.assertEqual(len(stored.state.context), 1)

    async def test_get_latest_and_linear_history(self) -> None:
        """list walks head -> root via parents (newest first)."""
        cpr = MemoryCheckpointer()
        a = _cp("t", step=0)
        b = _cp("t", parent=a.checkpoint_id, step=1)
        c = _cp("t", parent=b.checkpoint_id, step=2)
        for cp in (a, b, c):
            await cpr.put(cp)

        latest = await cpr.get_latest("t")
        self.assertEqual(latest.checkpoint_id, c.checkpoint_id)

        hist = await cpr.list("t")
        self.assertEqual(
            [h.checkpoint_id for h in hist],
            [c.checkpoint_id, b.checkpoint_id, a.checkpoint_id],
        )

    async def test_list_limit_and_from(self) -> None:
        """list honours limit and an explicit starting checkpoint."""
        cpr = MemoryCheckpointer()
        a = _cp("t", step=0)
        b = _cp("t", parent=a.checkpoint_id, step=1)
        c = _cp("t", parent=b.checkpoint_id, step=2)
        for cp in (a, b, c):
            await cpr.put(cp)

        limited = await cpr.list("t", limit=2)
        self.assertEqual(
            [h.checkpoint_id for h in limited],
            [c.checkpoint_id, b.checkpoint_id],
        )

        from_b = await cpr.list("t", from_checkpoint_id=b.checkpoint_id)
        self.assertEqual(
            [h.checkpoint_id for h in from_b],
            [b.checkpoint_id, a.checkpoint_id],
        )

    async def test_fork_keeps_old_branch(self) -> None:
        """Forking from an old checkpoint keeps siblings retrievable; the
        main line follows the new head."""
        cpr = MemoryCheckpointer()
        a = _cp("t", step=0)
        b = _cp("t", parent=a.checkpoint_id, step=1)
        c = _cp("t", parent=b.checkpoint_id, step=2)
        for cp in (a, b, c):
            await cpr.put(cp)

        # Fork off a: new child d with parent a.
        d = _cp("t", parent=a.checkpoint_id, step=1)
        await cpr.put(d)

        # Head/main line now follows d.
        self.assertEqual((await cpr.get_latest("t")).checkpoint_id,
                         d.checkpoint_id)
        self.assertEqual(
            [h.checkpoint_id for h in await cpr.list("t")],
            [d.checkpoint_id, a.checkpoint_id],
        )

        # Old branch still fully retrievable.
        self.assertIsNotNone(await cpr.get("t", b.checkpoint_id))
        self.assertIsNotNone(await cpr.get("t", c.checkpoint_id))
        self.assertEqual(
            [h.checkpoint_id for h in
             await cpr.list("t", from_checkpoint_id=c.checkpoint_id)],
            [c.checkpoint_id, b.checkpoint_id, a.checkpoint_id],
        )

    async def test_delete_thread(self) -> None:
        """delete_thread clears all checkpoints for the thread."""
        cpr = MemoryCheckpointer()
        a = _cp("t")
        await cpr.put(a)
        await cpr.delete_thread("t")
        self.assertIsNone(await cpr.get_latest("t"))
        self.assertEqual(await cpr.list("t"), [])
        self.assertIsNone(await cpr.get("t", a.checkpoint_id))

    async def test_retention_max_per_thread(self) -> None:
        """max_per_thread drops the oldest checkpoints, keeps the head."""
        cpr = MemoryCheckpointer(
            retention=RetentionPolicy(max_per_thread=2),
        )
        a = _cp("t", step=0)
        b = _cp("t", parent=a.checkpoint_id, step=1)
        c = _cp("t", parent=b.checkpoint_id, step=2)
        for cp in (a, b, c):
            await cpr.put(cp)

        self.assertIsNone(await cpr.get("t", a.checkpoint_id))
        self.assertIsNotNone(await cpr.get("t", b.checkpoint_id))
        self.assertIsNotNone(await cpr.get("t", c.checkpoint_id))

    async def test_retention_ttl(self) -> None:
        """ttl_seconds drops checkpoints older than the TTL on a later put,
        while never dropping the just-put head."""
        cpr = MemoryCheckpointer(
            retention=RetentionPolicy(ttl_seconds=100),
        )
        old = Checkpoint(
            thread_id="t",
            state=AgentState(session_id="t"),
            created_at=time.time() - 1000,
        )
        await cpr.put(old)
        # old is head at this moment -> survives its own put.
        self.assertIsNotNone(await cpr.get("t", old.checkpoint_id))

        fresh = Checkpoint(
            thread_id="t",
            parent_checkpoint_id=old.checkpoint_id,
            state=AgentState(session_id="t"),
        )
        await cpr.put(fresh)

        self.assertIsNone(await cpr.get("t", old.checkpoint_id))
        self.assertIsNotNone(await cpr.get("t", fresh.checkpoint_id))
