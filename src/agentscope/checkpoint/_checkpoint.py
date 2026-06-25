# -*- coding: utf-8 -*-
"""The checkpoint data models for the backtracking capability."""
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from .._utils._common import _generate_id
from ..message import Msg
from ..state import AgentState


class Checkpoint(BaseModel):
    """A single point-in-time snapshot of an agent run.

    Mirrors LangGraph's checkpoint: a checkpoint belongs to a thread
    (``thread_id`` == ``session_id``) and links to its predecessor via
    ``parent_checkpoint_id``, so checkpoints form a DAG. The default
    linear traversal (head -> root following parents) is the "linear
    backtracking" view; rewinding to an earlier checkpoint and continuing
    forks a new branch.

    The snapshot is self-contained: it carries a full :class:`AgentState`
    and (optionally) a full message-list snapshot, so a forked branch is
    isolated from its siblings.
    """

    checkpoint_id: str = Field(default_factory=_generate_id)
    """Unique id of this checkpoint."""

    thread_id: str
    """The thread this checkpoint belongs to (== ``session_id``)."""

    parent_checkpoint_id: str | None = None
    """The id of the parent checkpoint; ``None`` for the root."""

    step: int = 0
    """The super-step index (the agent's ``cur_iter``) this snapshot was
    taken at."""

    created_at: float = Field(default_factory=time.time)
    """The wall-clock creation time (epoch seconds)."""

    source: Literal["iteration", "fork", "manual"] = "iteration"
    """What triggered this checkpoint."""

    state: AgentState
    """The full agent state snapshot."""

    messages: list[Msg] = Field(default_factory=list)
    """An optional full snapshot of the display message list. Populated by
    integrations (e.g. the app layer) that keep messages separate from
    ``state.context``; empty for bare core usage."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Free-form metadata (trigger info, writes, etc.)."""


class RetentionPolicy(BaseModel):
    """How many checkpoints to keep per thread.

    Both limits default to ``None`` (keep everything). A checkpointer
    applies the policy on ``put``.
    """

    max_per_thread: int | None = None
    """Keep at most this many checkpoints per thread; ``None`` = unlimited."""

    ttl_seconds: float | None = None
    """Drop checkpoints older than this many seconds; ``None`` = no TTL."""
