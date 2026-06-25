# -*- coding: utf-8 -*-
"""Tests for the additive AgentState.checkpoint_id head pointer."""
from unittest import TestCase

from agentscope.state import AgentState


class AgentStateCheckpointFieldTest(TestCase):
    """The checkpoint_id field is additive and optional."""

    def test_default_is_none(self) -> None:
        """A fresh state has no checkpoint head."""
        self.assertIsNone(AgentState().checkpoint_id)

    def test_set_and_roundtrip(self) -> None:
        """The head pointer survives a dump/validate round-trip."""
        state = AgentState(session_id="t")
        state.checkpoint_id = "cp-1"
        restored = AgentState.model_validate(state.model_dump())
        self.assertEqual(restored.checkpoint_id, "cp-1")

    def test_back_compat_missing_field(self) -> None:
        """Old persisted state dicts without the field still validate."""
        data = AgentState(session_id="t").model_dump()
        data.pop("checkpoint_id", None)
        restored = AgentState.model_validate(data)
        self.assertIsNone(restored.checkpoint_id)
