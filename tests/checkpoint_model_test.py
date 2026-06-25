# -*- coding: utf-8 -*-
"""Tests for the Checkpoint and RetentionPolicy data models."""
from unittest import TestCase

from agentscope.checkpoint import Checkpoint, RetentionPolicy
from agentscope.state import AgentState


class CheckpointModelTest(TestCase):
    """The Checkpoint data model tests."""

    def test_defaults(self) -> None:
        """A checkpoint fills sensible defaults around the required fields."""
        state = AgentState(session_id="sess-1")
        cp = Checkpoint(thread_id="sess-1", state=state)

        self.assertTrue(cp.checkpoint_id)
        self.assertEqual(cp.thread_id, "sess-1")
        self.assertIsNone(cp.parent_checkpoint_id)
        self.assertEqual(cp.step, 0)
        self.assertEqual(cp.source, "iteration")
        self.assertIsInstance(cp.created_at, float)
        self.assertEqual(cp.messages, [])
        self.assertEqual(cp.metadata, {})
        self.assertEqual(cp.state.session_id, "sess-1")

    def test_unique_checkpoint_ids(self) -> None:
        """Each checkpoint gets its own id by default."""
        state = AgentState()
        a = Checkpoint(thread_id="t", state=state)
        b = Checkpoint(thread_id="t", state=state)
        self.assertNotEqual(a.checkpoint_id, b.checkpoint_id)

    def test_explicit_fields(self) -> None:
        """Explicit values for parent/step/source are preserved."""
        state = AgentState()
        cp = Checkpoint(
            thread_id="t",
            state=state,
            parent_checkpoint_id="parent-1",
            step=3,
            source="fork",
            metadata={"k": "v"},
        )
        self.assertEqual(cp.parent_checkpoint_id, "parent-1")
        self.assertEqual(cp.step, 3)
        self.assertEqual(cp.source, "fork")
        self.assertEqual(cp.metadata, {"k": "v"})

    def test_roundtrip_serialisation(self) -> None:
        """A checkpoint survives a pydantic dump/validate round-trip."""
        state = AgentState(session_id="sess-2")
        cp = Checkpoint(thread_id="sess-2", state=state, step=1)
        restored = Checkpoint.model_validate(cp.model_dump())
        self.assertEqual(restored.checkpoint_id, cp.checkpoint_id)
        self.assertEqual(restored.thread_id, "sess-2")
        self.assertEqual(restored.step, 1)
        self.assertEqual(restored.state.session_id, "sess-2")


class RetentionPolicyTest(TestCase):
    """The RetentionPolicy data model tests."""

    def test_defaults_keep_all(self) -> None:
        """Default policy keeps everything (no cap, no TTL)."""
        rp = RetentionPolicy()
        self.assertIsNone(rp.max_per_thread)
        self.assertIsNone(rp.ttl_seconds)

    def test_explicit_limits(self) -> None:
        """Explicit limits are stored as given."""
        rp = RetentionPolicy(max_per_thread=5, ttl_seconds=60.0)
        self.assertEqual(rp.max_per_thread, 5)
        self.assertEqual(rp.ttl_seconds, 60.0)
