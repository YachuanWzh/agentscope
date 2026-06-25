# Backtracking (Checkpoint / Time-Travel) Demo

A minimal, **zero-setup** demo of AgentScope's core backtracking capability —
inspired by LangGraph's checkpointer. Every agent run is bound to a
`session_id` (the LangGraph `thread_id` equivalent), and you can rewind to any
earlier point and continue, forking a new branch.

## What it shows

1. **Auto-checkpointing** — give an `Agent` a `MemoryCheckpointer` and a
   checkpoint is saved automatically after every ReAct iteration.
2. **Linear history** — `agent.get_state_history()` returns the main line
   (newest first), i.e. the linear backtracking view.
3. **Rewind** — `agent.rewind(checkpoint_id)` restores the agent's full state
   (context, iteration counter, permissions, …) to that checkpoint.
4. **Fork** — replying after a rewind branches off the rewound point; the
   previously-current branch is preserved and still retrievable by id.

## Run it

No API key required — the demo uses a tiny offline scripted model.

```bash
# from the repo root
PYTHONPATH=src python examples/checkpoint_backtracking/main.py
```

(If you have the package installed, plain `python main.py` works too.)

### Expected output (ids will differ)

```
session_id (thread_id) = 7f86c161bd2c46bbae9744a16370d545

[run 1] final reply: Plan complete: research -> draft.

[run 1] checkpoint history (newest first):
  HEAD -> step=2  id=87d277ef  parent=d5c22cbf  src=iteration
          step=1  id=d5c22cbf  parent=-         src=iteration

Rewinding to the 'research' checkpoint (id=d5c22cbf, step=1) ...
  agent head is now: d5c22cbf
  restored cur_iter: 1

[run 2] final reply: Plan complete: research -> outline.

[run 2] new main line after the fork:
  HEAD -> step=1  id=e51c1f29  parent=d5c22cbf  src=iteration
          step=1  id=d5c22cbf  parent=-         src=iteration

Abandoned 'draft' branch still retrievable by id (87d277ef): True
```

## The API in one glance

```python
from agentscope.agent import Agent
from agentscope.checkpoint import MemoryCheckpointer

agent = Agent(..., checkpointer=MemoryCheckpointer())

await agent.reply(user_msg)              # checkpoints saved per iteration
history = await agent.get_state_history()  # linear main line, newest first
await agent.rewind(history[-1].checkpoint_id)  # time-travel to an earlier point
await agent.reply(other_msg)             # forks a new branch from there
```

## Using a real LLM

Swap the offline `ScriptedChatModel` for a real model and drop the
`model.script(...)` calls — checkpointing/rewind/fork behave identically. For
example, with DashScope:

```python
import os
from agentscope.model import DashScopeChatModel
from agentscope.credential import DashScopeCredential

model = DashScopeChatModel(
    credential=DashScopeCredential(api_key=os.environ["DASHSCOPE_API_KEY"]),
    model="qwen-max",
)
agent = Agent(name="Planner", system_prompt="...", model=model,
              checkpointer=MemoryCheckpointer())
await agent.reply(UserMsg(name="user", content="Make a 2-step plan."))
```

## Notes

- `MemoryCheckpointer` is process-local (great for scripts/tests). Pass a
  `RetentionPolicy(max_per_thread=..., ttl_seconds=...)` to cap growth; the
  default keeps everything.
- A persistent (e.g. Redis-backed) checkpointer for the app/service layer is a
  planned follow-up — see
  `.claude/superharness/specs/2026-06-25-agentscope-checkpoint-backtracking.md`.
