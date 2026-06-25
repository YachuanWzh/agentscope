# -*- coding: utf-8 -*-
"""Tests for Agent <-> checkpointer integration and time-travel API."""
from typing import Any
from unittest import IsolatedAsyncioTestCase

from utils import MockModel

from agentscope.agent import Agent
from agentscope.checkpoint import MemoryCheckpointer
from agentscope.model import ChatResponse
from agentscope.tool import ToolBase, Toolkit, ToolChunk
from agentscope.permission import (
    PermissionDecision,
    PermissionBehavior,
    PermissionContext,
)
from agentscope.message import TextBlock, ToolCallBlock, UserMsg


class _EchoTool(ToolBase):
    """A trivial sequential tool used to drive multi-iteration replies."""

    name: str = "echo_tool"
    description: str = "Echo the input."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    }
    is_concurrency_safe: bool = False
    is_read_only: bool = True
    is_external_tool: bool = False
    is_mcp: bool = False

    async def check_permissions(
        self,
        tool_input: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionDecision:
        """Always allow."""
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            decision_reason="ok",
            message="ok",
        )

    # pylint: disable=redefined-builtin
    async def __call__(self, input: str, **kwargs: Any) -> ToolChunk:
        """Echo back the input."""
        return ToolChunk(content=[TextBlock(text=f"echo: {input}")])


def _tool_call(i: int) -> ToolCallBlock:
    """Build an echo tool-call block."""
    return ToolCallBlock(
        id=f"tc{i}",
        name="echo_tool",
        input='{"input": "x"}',
    )


def _two_tool_rounds_then_text(model: MockModel) -> None:
    """Two tool-calling iterations followed by a final text answer."""
    model.set_responses(
        [
            [ChatResponse(content=[_tool_call(1)], is_last=True)],
            [ChatResponse(content=[_tool_call(2)], is_last=True)],
            [ChatResponse(content=[TextBlock(text="done")], is_last=True)],
        ],
    )


class AgentCheckpointIntegrationTest(IsolatedAsyncioTestCase):
    """Auto-checkpointing per ReAct iteration."""

    def _make_agent(
        self,
        checkpointer: MemoryCheckpointer | None,
    ) -> tuple[Agent, MockModel]:
        """Build an agent wired with the echo tool and given checkpointer."""
        model = MockModel()
        agent = Agent(
            name="Friday",
            system_prompt="You are helpful.",
            model=model,
            toolkit=Toolkit(tools=[_EchoTool()]),
            checkpointer=checkpointer,
        )
        return agent, model

    async def test_checkpoint_per_iteration(self) -> None:
        """Each completed ReAct iteration produces one checkpoint with a
        correct parent chain and step, and updates the head pointer."""
        cpr = MemoryCheckpointer()
        agent, model = self._make_agent(cpr)
        _two_tool_rounds_then_text(model)

        await agent.reply(UserMsg(name="user", content="hi"))
        thread = agent.state.session_id

        hist = await cpr.list(thread)
        self.assertEqual(len(hist), 2)
        # newest-first: step 2 then step 1
        self.assertEqual([h.step for h in hist], [2, 1])
        # parent chain: oldest has no parent, newest points to oldest
        self.assertIsNone(hist[1].parent_checkpoint_id)
        self.assertEqual(
            hist[0].parent_checkpoint_id,
            hist[1].checkpoint_id,
        )
        # head pointer on the live state matches the latest checkpoint
        self.assertEqual(agent.state.checkpoint_id, hist[0].checkpoint_id)
        # snapshot self-id is consistent
        self.assertEqual(hist[0].state.checkpoint_id, hist[0].checkpoint_id)

    async def test_no_checkpointer_is_noop(self) -> None:
        """Without a checkpointer the head pointer stays None (no behaviour
        change)."""
        agent, model = self._make_agent(None)
        self.assertIsNone(agent.checkpointer)
        _two_tool_rounds_then_text(model)

        await agent.reply(UserMsg(name="user", content="hi"))
        self.assertIsNone(agent.state.checkpoint_id)
