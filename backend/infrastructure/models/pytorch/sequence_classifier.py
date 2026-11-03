"""GRU, LSTM and BiLSTM classifiers over one shared implementation.

The three architectures differ in exactly one place: which recurrent module
encodes the sequence, and therefore how wide its final representation is.
Everything else — padding-aware packing, how the final state is extracted, the
classification head, the fact that the output is a logit — is defined once in
:class:`RecurrentSequenceClassifier` and inherited unchanged.

That is not only tidier. The project's central comparison is GRU against LSTM
against BiLSTM on identical data, and three independently written models would
make any measured difference partly an artefact of three different pieces of code.
Here the subclasses declare an encoder and nothing else, so the difference between
the baselines is the difference between the architectures.

Two specifics worth stating:

**Padding is packed, not read.** Sequences are right-padded with zeros, and
:func:`~torch.nn.utils.rnn.pack_padded_sequence` is given the true lengths so the
encoder stops at the last real transaction. This matters most for the
bidirectional model: without packing its backward pass would begin inside the
padding and the "final" backward state would summarise zeros.

**The output is a logit.** No sigmoid anywhere in this module. The loss is
``BCEWithLogitsLoss``, which is numerically stable precisely because it fuses the
sigmoid, and probabilities are produced once, at evaluation.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture


class RecurrentSequenceClassifier(nn.Module):
    """A recurrent encoder plus a single-logit head.

    Attributes:
        feature_dimension: Width of the input feature axis.
        hidden_size: Hidden width per direction.
        num_layers: Stacked recurrent layers.
    """

    def __init__(
        self,
        *,
        feature_dimension: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Build the encoder and the head.

        Args:
            feature_dimension: Input width, which is the sequence provider's
                feature dimension.
            hidden_size: Hidden units per direction.
            num_layers: Stacked layers.
            dropout: Dropout between layers. PyTorch ignores it for a single
                layer, so callers should pass
                :attr:`TrainingConfig.effective_dropout` to keep the reported
                configuration honest.

        Raises:
            ValueError: A dimension is not positive.
        """
        super().__init__()
        if feature_dimension <= 0:
            raise ValueError(
                f"feature_dimension must be positive; got {feature_dimension!r}"
            )
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive; got {hidden_size!r}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive; got {num_layers!r}")

        self.feature_dimension = feature_dimension
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = self._build_encoder(dropout)
        self.head = nn.Linear(self.representation_size, 1)

    # --- Subclass contract ---------------------------------------------------

    @abstractmethod
    def _build_encoder(self, dropout: float) -> nn.RNNBase:
        """Return the recurrent module for this architecture."""

    @property
    @abstractmethod
    def directions(self) -> int:
        """1 for a unidirectional encoder, 2 for a bidirectional one."""

    @property
    def representation_size(self) -> int:
        """Width of the vector handed to the head.

        Doubled for a bidirectional encoder, because its final representation is
        the forward and backward states concatenated. Derived here rather than
        written out per subclass so a bidirectional model cannot end up with a head
        sized for one direction — a mistake that trains without complaint and
        silently discards half the encoder.
        """
        return self.hidden_size * self.directions

    # --- Forward -------------------------------------------------------------

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        """Encode padded sequences and return one logit each.

        Args:
            features: ``(batch, sequence_length, feature_dimension)`` float
                tensor, right-padded with zeros.
            lengths: ``(batch,)`` true lengths, each at least 1.

        Returns:
            ``(batch,)`` logits. Not probabilities: no sigmoid is applied here.
        """
        packed = pack_padded_sequence(
            features,
            # pack_padded_sequence requires lengths on the CPU as int64, whatever
            # device the features are on.
            lengths.detach().to(device="cpu", dtype=torch.int64),
            batch_first=True,
            # The batch arrives in dataset or shuffled order, not sorted by length;
            # sorting it here would break the positional alignment between
            # predictions and labels that evaluation depends on.
            enforce_sorted=False,
        )
        _, state = self.encoder(packed)
        # Annotated rather than returned inline: `nn.Module.__call__` is typed as
        # returning `Any`, and naming the type here keeps the port's contract checked.
        logits: Tensor = self.head(self._final_representation(state))
        return logits.squeeze(-1)

    def _final_representation(self, state: Tensor | tuple[Tensor, ...]) -> Tensor:
        """Extract the last layer's final hidden state as a flat vector.

        ``state`` is the hidden state for a GRU and a ``(hidden, cell)`` pair for an
        LSTM; only the hidden part is used. Its shape is
        ``(num_layers * directions, batch, hidden_size)``, ordered so that the last
        layer's directions are the final one or two entries.
        """
        hidden = state[0] if isinstance(state, tuple) else state
        if self.directions == 1:
            return hidden[-1]
        # Forward then backward, in that order, so the head's input layout is
        # stable and a saved checkpoint stays interpretable.
        return torch.cat((hidden[-2], hidden[-1]), dim=1)

    def parameter_count(self) -> int:
        """Trainable parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class GruClassifier(RecurrentSequenceClassifier):
    """Gated recurrent unit encoder, unidirectional."""

    def _build_encoder(self, dropout: float) -> nn.RNNBase:
        """Build a unidirectional GRU."""
        return nn.GRU(
            input_size=self.feature_dimension,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

    @property
    def directions(self) -> int:
        """One."""
        return 1


class LstmClassifier(RecurrentSequenceClassifier):
    """Long short-term memory encoder, unidirectional."""

    def _build_encoder(self, dropout: float) -> nn.RNNBase:
        """Build a unidirectional LSTM."""
        return nn.LSTM(
            input_size=self.feature_dimension,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

    @property
    def directions(self) -> int:
        """One."""
        return 1


class BiLstmClassifier(RecurrentSequenceClassifier):
    """Bidirectional LSTM encoder.

    Reads each account's window forwards and backwards, so the representation of a
    transaction can be informed by the whole window rather than only its prefix.
    Note what this does and does not mean for causality: the window still ends at
    the transaction being classified, so the backward pass reads that account's
    *earlier* transactions in reverse — never anything from the future.
    """

    def _build_encoder(self, dropout: float) -> nn.RNNBase:
        """Build a bidirectional LSTM."""
        return nn.LSTM(
            input_size=self.feature_dimension,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

    @property
    def directions(self) -> int:
        """Two, which doubles :attr:`representation_size`."""
        return 2


_CLASSIFIERS: dict[ModelArchitecture, type[RecurrentSequenceClassifier]] = {
    ModelArchitecture.GRU: GruClassifier,
    ModelArchitecture.LSTM: LstmClassifier,
    ModelArchitecture.BILSTM: BiLstmClassifier,
}


def build_classifier(
    config: TrainingConfig, feature_dimension: int
) -> nn.Module:
    """Construct the model one configuration describes.

    Args:
        config: Supplies the architecture and its hyperparameters.
        feature_dimension: Input width, taken from the data rather than assumed.

    Returns:
        An untrained model on the CPU.
    """
    if config.architecture not in _CLASSIFIERS:
        raise ValueError(
            f"Unsupported architecture '{config.architecture}'. "
            f"Active supported architectures are: {list(_CLASSIFIERS.keys())}"
        )

    classifier = _CLASSIFIERS[config.architecture]
    return classifier(
        feature_dimension=feature_dimension,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        # The effective value, so a single-layer model is not recorded as
        # regularised by a dropout PyTorch silently ignored.
        dropout=config.effective_dropout,
    )
