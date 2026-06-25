# -*- coding: utf-8 -*-
"""A self-contained demo of AgentScope's backtracking (checkpoint /
time-travel) capability.

It shows the four core moves of the feature, all keyed by ``session_id``
(the LangGraph ``thread_id`` equivalent):

1. Run an agent with a ``MemoryCheckpointer`` -> a checkpoint is saved
   automatically after every ReAct iteration.
2. ``get_state_history()`` -> the linear backtracking history (newest first).
3. ``rewind(checkpoint_id)`` -> restore the agent to an earlier checkpoint.
4. Reply again -> a new branch forks off the rewound point, while the
   previously-current branch stays retrievable.

To keep the demo runnable with **zero API keys**, it drives the agent with a
tiny offline ``ScriptedChatModel`` that returns canned responses. To try it
against a real LLM instead, see the note at the bottom of this file and the
README.
"""
import asyncio
from typing import Any, AsyncGenerator

from pydantic import BaseModel

from agentscope.agent import Agent
from agentscope.checkpoint import MemoryCheckpointer
from agentscope.credential import CredentialBase
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.tool import ToolBase, Toolkit, ToolChunk
from agentscope.permission import (
    PermissionContext,
    PermissionDecision,
    PermissionBehavior,
)
from agentscope.message import TextBlock, ToolCallBlock, UserMsg


# ---------------------------------------------------------------------------
# A tiny offline model so the demo runs without any credentials.
# Replace this with a real model (e.g. DashScopeChatModel) for live use.
# ---------------------------------------------------------------------------
class _ScriptedCredential(CredentialBase):
    """A placeholder credential for the offline scripted model."""

    @classmethod
    def get_chat_model_class(cls) -> type[ChatModelBase]:
        """Return the scripted model class."""
        return ScriptedChatModel


class ScriptedChatModel(ChatModelBase):
    """A non-streaming model that replays a fixed list of responses."""

    class Parameters(BaseModel):
        """No tunable parameters for the scripted model."""

    def __init__(self) -> None:
        """Initialise the scripted model with an empty script."""
        super().__init__(
            credential=_ScriptedCredential(),
            model="scripted-model",
            parameters=ScriptedChatModel.Parameters(),
            stream=False,
            context_size=8192,
        )
        self._responses: list[ChatResponse] = []
        self._cursor = 0

    def script(self, responses: list[ChatResponse]) -> None:
        """Load the responses for the next reply and reset the cursor."""
        self._responses = list(responses)
        self._cursor = 0

    async def _call_api(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """Return the next scripted response."""
        response = self._responses[self._cursor]
        self._cursor += 1
        return response


# ---------------------------------------------------------------------------
# A trivial tool so each reply runs multiple ReAct iterations (= checkpoints).
# ---------------------------------------------------------------------------
class RecordStepTool(ToolBase):
    """Record one completed step of a plan."""

    name: str = "record_step"
    description: str = "Record a completed step of the plan."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "step": {"type": "string", "description": "The step name."},
        },
        "required": ["step"],
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
        """Always allow in this demo."""
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            decision_reason="demo",
            message="demo",
        )

    async def __call__(self, step: str, **kwargs: Any) -> ToolChunk:
        """Record the step."""
        return ToolChunk(content=[TextBlock(text=f"recorded: {step}")])


def _tool_call(call_id: str, step: str) -> ChatResponse:
    """A model response that calls ``record_step`` with ``step``."""
    return ChatResponse(
        content=[
            ToolCallBlock(
                id=call_id,
                name="record_step",
                input=f'{{"step": "{step}"}}',
            ),
        ],
        is_last=True,
    )


def _final(text: str) -> ChatResponse:
    """A model response that ends the reply with plain text."""
    return ChatResponse(content=[TextBlock(text=text)], is_last=True)


def _print_history(label: str, history: list) -> None:
    """Pretty-print a checkpoint history (newest first)."""
    print(f"\n{label}")
    for cp in history:
        marker = "HEAD ->" if cp is history[0] else "       "
        print(
            f"  {marker} step={cp.step}  id={cp.checkpoint_id[:8]}  "
            f"parent={(cp.parent_checkpoint_id or '-')[:8]}  "
            f"src={cp.source}",
        )


async def main() -> None:
    """Run the backtracking demo end to end."""
    checkpointer = MemoryCheckpointer()
    model = ScriptedChatModel()
    agent = Agent(
        name="Planner",
        system_prompt="You are a planning assistant.",
        model=model,
        toolkit=Toolkit(tools=[RecordStepTool()]),
        checkpointer=checkpointer,
    )
    thread_id = agent.state.session_id
    print(f"session_id (thread_id) = {thread_id}")

    # --- 1. First run: two tool iterations, then a final answer ----------
    model.script(
        [
            _tool_call("c1", "research"),
            _tool_call("c2", "draft"),
            _final("Plan complete: research -> draft."),
        ],
    )
    reply1 = await agent.reply(UserMsg(name="user", content="Make a 2-step plan."))
    print(f"\n[run 1] final reply: {reply1.get_text_content()}")

    # --- 2. Inspect the linear backtracking history ----------------------
    history = await agent.get_state_history()
    _print_history("[run 1] checkpoint history (newest first):", history)

    # --- 3. Rewind to the earlier 'research' checkpoint ------------------
    research_cp = history[-1]  # oldest = after step 1 (research)
    draft_cp = history[0]      # newest = after step 2 (draft)
    print(
        f"\nRewinding to the 'research' checkpoint "
        f"(id={research_cp.checkpoint_id[:8]}, step={research_cp.step}) ...",
    )
    await agent.rewind(research_cp.checkpoint_id)
    print(f"  agent head is now: {agent.state.checkpoint_id[:8]}")
    print(f"  restored cur_iter: {agent.state.cur_iter}")

    # --- 4. Reply again -> forks a new branch off 'research' -------------
    model.script(
        [
            _tool_call("c3", "outline"),
            _final("Plan complete: research -> outline."),
        ],
    )
    reply2 = await agent.reply(
        UserMsg(name="user", content="Actually, outline instead of draft."),
    )
    print(f"\n[run 2] final reply: {reply2.get_text_content()}")

    forked = await agent.get_state_history()
    _print_history("[run 2] new main line after the fork:", forked)

    # The abandoned 'draft' branch is still fully retrievable by id.
    abandoned = await checkpointer.get(thread_id, draft_cp.checkpoint_id)
    print(
        f"\nAbandoned 'draft' branch still retrievable by id "
        f"({draft_cp.checkpoint_id[:8]}): {abandoned is not None}",
    )
    print(
        "  -> linear backtracking gives you the current branch, while the "
        "full fork tree stays in the checkpointer.",
    )


# ---------------------------------------------------------------------------
# Want to run this against a real LLM? Swap the two lines in ``main`` that
# build ``model`` / call ``model.script(...)`` for a real model and a real
# user prompt, e.g.:
#
#     from agentscope.model import DashScopeChatModel
#     from agentscope.credential import DashScopeCredential
#     model = DashScopeChatModel(
#         credential=DashScopeCredential(api_key=os.environ["DASHSCOPE_API_KEY"]),
#         model="qwen-max",
#     )
#
# Then just `await agent.reply(UserMsg(...))` normally (no `model.script`).
# Everything about checkpointing / rewind / fork works identically.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
