# -*- coding: utf-8 -*-
"""An in-memory checkpointer implementation."""
import time

from ._checkpoint import Checkpoint, RetentionPolicy
from ._checkpointer_base import CheckpointerBase


class MemoryCheckpointer(CheckpointerBase):
    """A process-local checkpointer backed by plain dicts.

    Intended for bare SDK usage and tests, mirroring the in-memory message
    bus. Checkpoints are deep-copied on store and on retrieval so the
    store stays isolated from caller-side mutation.

    Args:
        retention (`RetentionPolicy | None`, optional):
            How many checkpoints to keep per thread. Defaults to keeping
            everything.
    """

    def __init__(self, retention: RetentionPolicy | None = None) -> None:
        """Initialise the in-memory checkpointer."""
        self._retention = retention or RetentionPolicy()
        # thread_id -> {checkpoint_id: Checkpoint}
        self._store: dict[str, dict[str, Checkpoint]] = {}
        # thread_id -> head checkpoint_id
        self._head: dict[str, str] = {}

    async def put(self, checkpoint: Checkpoint) -> str:
        """Store a checkpoint (deep-copied) and make it the head."""
        snapshot = checkpoint.model_copy(deep=True)
        thread = snapshot.thread_id
        self._store.setdefault(thread, {})[snapshot.checkpoint_id] = snapshot
        self._head[thread] = snapshot.checkpoint_id
        self._prune(thread)
        return snapshot.checkpoint_id

    async def get(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> Checkpoint | None:
        """Fetch a single checkpoint by id (deep-copied)."""
        cp = self._store.get(thread_id, {}).get(checkpoint_id)
        return cp.model_copy(deep=True) if cp is not None else None

    async def get_latest(self, thread_id: str) -> Checkpoint | None:
        """Return the thread's head checkpoint (deep-copied)."""
        head_id = self._head.get(thread_id)
        if head_id is None:
            return None
        return await self.get(thread_id, head_id)

    async def list(
        self,
        thread_id: str,
        *,
        from_checkpoint_id: str | None = None,
        limit: int | None = None,
    ) -> list[Checkpoint]:
        """Walk parents from the start point to the root, newest first."""
        thread = self._store.get(thread_id, {})
        cur = from_checkpoint_id or self._head.get(thread_id)

        result: list[Checkpoint] = []
        while cur is not None:
            cp = thread.get(cur)
            if cp is None:
                break
            result.append(cp.model_copy(deep=True))
            if limit is not None and len(result) >= limit:
                break
            cur = cp.parent_checkpoint_id
        return result

    async def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread."""
        self._store.pop(thread_id, None)
        self._head.pop(thread_id, None)

    def _prune(self, thread_id: str) -> None:
        """Apply the retention policy to a thread, never dropping the head."""
        thread = self._store.get(thread_id)
        if not thread:
            return
        head_id = self._head.get(thread_id)

        # TTL: drop checkpoints older than ttl_seconds (except the head).
        ttl = self._retention.ttl_seconds
        if ttl is not None:
            cutoff = time.time() - ttl
            for cpid in [
                cpid
                for cpid, cp in thread.items()
                if cpid != head_id and cp.created_at < cutoff
            ]:
                del thread[cpid]

        # Cap: keep at most max_per_thread, dropping oldest (except the head).
        cap = self._retention.max_per_thread
        if cap is not None and len(thread) > cap:
            # Oldest first by created_at; never evict the head.
            evictable = sorted(
                (cp for cpid, cp in thread.items() if cpid != head_id),
                key=lambda cp: cp.created_at,
            )
            overflow = len(thread) - cap
            for cp in evictable[:overflow]:
                del thread[cp.checkpoint_id]
