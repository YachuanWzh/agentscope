# -*- coding: utf-8 -*-
"""The checkpointer abstract base class."""
from abc import ABC, abstractmethod

from ._checkpoint import Checkpoint


class CheckpointerBase(ABC):
    """Persist and retrieve :class:`Checkpoint` snapshots for a thread.

    Mirrors LangGraph's ``BaseCheckpointSaver``: implementations store
    checkpoints keyed by ``thread_id`` (== ``session_id``) and expose a
    linear history (head -> root following ``parent_checkpoint_id``), which
    is the default "linear backtracking" view over a possibly-forked DAG.
    """

    @abstractmethod
    async def put(self, checkpoint: Checkpoint) -> str:
        """Store a checkpoint and make it the thread's new head.

        Args:
            checkpoint (`Checkpoint`):
                The checkpoint to store. Implementations must snapshot it
                (deep copy) so later caller mutations do not leak in.

        Returns:
            `str`: The stored checkpoint's id.
        """

    @abstractmethod
    async def get(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> Checkpoint | None:
        """Fetch a single checkpoint by id.

        Args:
            thread_id (`str`): The thread the checkpoint belongs to.
            checkpoint_id (`str`): The checkpoint id.

        Returns:
            `Checkpoint | None`: The checkpoint, or ``None`` if not found.
        """

    @abstractmethod
    async def get_latest(self, thread_id: str) -> Checkpoint | None:
        """Return the thread's head (most recently put) checkpoint.

        Args:
            thread_id (`str`): The thread id.

        Returns:
            `Checkpoint | None`: The head checkpoint, or ``None`` if the
            thread has no checkpoints.
        """

    @abstractmethod
    async def list(
        self,
        thread_id: str,
        *,
        from_checkpoint_id: str | None = None,
        limit: int | None = None,
    ) -> list[Checkpoint]:
        """Return the linear history for a thread, newest first.

        Walks ``parent_checkpoint_id`` links starting from
        ``from_checkpoint_id`` (or the head when ``None``) back towards the
        root.

        Args:
            thread_id (`str`): The thread id.
            from_checkpoint_id (`str | None`, optional): Start the walk at
                this checkpoint instead of the head.
            limit (`int | None`, optional): Cap the number returned.

        Returns:
            `list[Checkpoint]`: Checkpoints from the start point to the
            root, newest first.
        """

    @abstractmethod
    async def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread.

        Args:
            thread_id (`str`): The thread id.
        """
