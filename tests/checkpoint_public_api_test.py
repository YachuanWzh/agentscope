# -*- coding: utf-8 -*-
"""Locks the public import surface of the checkpoint subpackage."""
from unittest import TestCase

import agentscope.checkpoint as cp


class CheckpointPublicApiTest(TestCase):
    """The checkpoint subpackage exposes its public names."""

    def test_public_names(self) -> None:
        """All documented names are importable and listed in __all__."""
        for name in (
            "Checkpoint",
            "RetentionPolicy",
            "CheckpointerBase",
            "MemoryCheckpointer",
        ):
            self.assertIn(name, cp.__all__)
            self.assertTrue(hasattr(cp, name))
