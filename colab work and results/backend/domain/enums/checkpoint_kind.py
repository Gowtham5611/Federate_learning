"""The two checkpoints an experiment writes, and why they are not the same file.

Under early stopping the best epoch is usually not the last one, so "the model"
is ambiguous unless the two are named and stored apart. Keeping them as distinct
enum members — and therefore distinct paths — is what stops a final-epoch write
from silently replacing the parameters whose validation score was reported.
"""

from __future__ import annotations

from enum import StrEnum


class CheckpointKind(StrEnum):
    """Which of an experiment's two checkpoints is meant."""

    BEST = "best"
    """Parameters from the epoch with the best validation score.

    The model that is evaluated on test, and the one a later phase would ship.
    """

    FINAL = "final"
    """Parameters after the last completed epoch.

    Kept for diagnosis — comparing it against ``BEST`` shows how far training
    drifted after its best point — and because resuming needs the last state, not
    the best one.
    """

    GLOBAL_ROUND = "global_round"
    """Global parameters aggregated after a federated round."""

    GLOBAL_BEST = "global_best"
    """Global parameters with the best centralized validation score."""
