"""The recurrent architectures this project compares.

Three members, deliberately. The research design fixes the comparison at GRU
versus LSTM versus BiLSTM, and every federated experiment in the later phases
trains *one* of these across all of its clients. Parameters from different
members are never averaged together, so this enum names three separate
experiment families rather than a menu of interchangeable options.
"""

from __future__ import annotations

from enum import StrEnum


class ModelArchitecture(StrEnum):
    """A sequence-classifier architecture.

    The value is the lowercase token used on the command line, in experiment
    identifiers and in result records, so one spelling serves the CLI, the
    filesystem and the stored results.
    """

    GRU = "gru"
    LSTM = "lstm"
    BILSTM = "bilstm"

    @property
    def is_bidirectional(self) -> bool:
        """Whether the encoder reads the sequence in both directions."""
        return self is ModelArchitecture.BILSTM

    @property
    def display_name(self) -> str:
        """Human-facing name, for logs and reports."""
        return _DISPLAY_NAMES[self]


_DISPLAY_NAMES: dict[ModelArchitecture, str] = {
    ModelArchitecture.GRU: "GRU",
    ModelArchitecture.LSTM: "LSTM",
    ModelArchitecture.BILSTM: "BiLSTM",
}

ORDERED_ARCHITECTURES: tuple[ModelArchitecture, ...] = (
    ModelArchitecture.GRU,
    ModelArchitecture.LSTM,
    ModelArchitecture.BILSTM,
)
"""Every architecture, in the order ``--model all`` trains them.

Fixed so that a run of all three is reproducible down to the order of the
experiments, and so the order carries no implication of ranking.
"""
