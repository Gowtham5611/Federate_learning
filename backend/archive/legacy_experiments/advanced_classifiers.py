"""Advanced Neural Classifiers for Federated AML Detection: TCN, Transformer Encoder, and Residual MLP.

All three architectures follow the established contract:
- Input: `features` (batch, seq_len, feature_dimension) and `lengths` (batch,) true sequence lengths.
- Output: `logits` (batch,) un-normalized log-odds for binary cross-entropy loss (BCEWithLogitsLoss).
- Causal guarantee: No lookahead or future information leakage. The transaction being classified
  is at index `lengths - 1`.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from application.dto.training import TrainingConfig


class ChausalConv1d(nn.Module):
    """1D Causal Convolution with dilation ensuring no future information leakage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,  # Manual causal padding
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x is (batch, channels, seq_len)
        # Pad left by self.padding, pad right by 0
        x_padded = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x_padded)


class TcnResidualBlock(nn.Module):
    """Residual block with two dilated causal convolutions, layer norms, and dropout."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = ChausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = ChausalConv1d(channels, channels, kernel_size, dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        res = x
        out = self.drop1(self.act1(self.norm1(self.conv1(x))))
        out = self.drop2(self.act2(self.norm2(self.conv2(out))))
        return out + res


class TcnClassifier(nn.Module):
    """Temporal Convolutional Network (TCN) with dilated causal residual blocks."""

    def __init__(
        self,
        *,
        feature_dimension: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_dimension = feature_dimension
        self.hidden_size = hidden_size
        self.input_proj = nn.Conv1d(feature_dimension, hidden_size, kernel_size=1)

        dilations = [2**i for i in range(num_layers)]  # e.g., [1, 2]
        self.blocks = nn.ModuleList([
            TcnResidualBlock(hidden_size, kernel_size=3, dilation=d, dropout=dropout)
            for d in dilations
        ])
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        # features: (batch, seq_len, feature_dim)
        # lengths: (batch,)
        batch_size, seq_len, _ = features.shape
        x = features.transpose(1, 2)  # (batch, feature_dim, seq_len)
        x = self.input_proj(x)        # (batch, hidden_size, seq_len)

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)  # (batch, seq_len, hidden_size)

        # Extract representation at index lengths - 1 for each batch item (the target transaction)
        idx = (lengths - 1).clamp(min=0, max=seq_len - 1).view(-1, 1, 1).expand(-1, 1, self.hidden_size)
        target_repr = torch.gather(x, dim=1, index=idx).squeeze(1)  # (batch, hidden_size)

        logits = self.head(target_repr).squeeze(-1)  # (batch,)
        return logits

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderClassifier(nn.Module):
    """Transformer Encoder classifier with multi-head self-attention and causal target extraction."""

    def __init__(
        self,
        *,
        feature_dimension: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_dimension = feature_dimension
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(feature_dimension, hidden_size)
        self.norm_in = nn.LayerNorm(hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=200)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        # features: (batch, seq_len, feature_dim)
        # lengths: (batch,)
        batch_size, seq_len, _ = features.shape
        device = features.device

        # Create padding mask: True where attention should be blocked
        arange_tensor = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        padding_mask = arange_tensor >= lengths.unsqueeze(1)               # (batch, seq_len)

        x = self.norm_in(self.input_proj(features))  # (batch, seq_len, hidden_size)
        x = self.pos_encoder(x)

        encoded = self.transformer_encoder(x, src_key_padding_mask=padding_mask)  # (batch, seq_len, hidden_size)

        # Extract target transaction token representation at lengths - 1
        idx = (lengths - 1).clamp(min=0, max=seq_len - 1).view(-1, 1, 1).expand(-1, 1, self.hidden_size)
        target_repr = torch.gather(encoded, dim=1, index=idx).squeeze(1)  # (batch, hidden_size)

        logits = self.head(target_repr).squeeze(-1)  # (batch,)
        return logits

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResNetBlock(nn.Module):
    """Dimension-preserving tabular residual block with LayerNorm, GELU, and Dropout."""

    def __init__(self, hidden_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        res = x
        out = self.act(self.linear1(self.norm1(x)))
        out = self.drop(out)
        out = self.linear2(self.norm2(out))
        return res + out


class ResidualMlpClassifier(nn.Module):
    """Tabular ResNet (Residual MLP) with historical sequence summary and residual blocks."""

    def __init__(
        self,
        *,
        feature_dimension: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_dimension = feature_dimension
        self.hidden_size = hidden_size

        # Combines the current transaction (at lengths - 1) with masked historical mean
        in_dim = feature_dimension * 2
        self.input_proj = nn.Linear(in_dim, hidden_size)
        self.norm_in = nn.LayerNorm(hidden_size)

        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_size, dropout=dropout) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        # features: (batch, seq_len, feature_dim)
        # lengths: (batch,)
        batch_size, seq_len, dim = features.shape
        device = features.device

        # 1. Target transaction at lengths - 1
        idx = (lengths - 1).clamp(min=0, max=seq_len - 1).view(-1, 1, 1).expand(-1, 1, dim)
        curr_tx = torch.gather(features, dim=1, index=idx).squeeze(1)  # (batch, feature_dim)

        # 2. Masked historical mean pooling (no lookahead, averages 0 .. lengths - 1)
        mask = (torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float().unsqueeze(-1)
        masked_sum = (features * mask).sum(dim=1)
        lengths_denom = lengths.float().clamp(min=1.0).unsqueeze(-1)
        hist_mean = masked_sum / lengths_denom  # (batch, feature_dim)

        # 3. Concatenate current transaction + historical context
        combined = torch.cat([curr_tx, hist_mean], dim=-1)  # (batch, 2 * feature_dim)

        x = self.norm_in(self.input_proj(combined))  # (batch, hidden_size)
        for block in self.blocks:
            x = block(x)

        logits = self.head(x).squeeze(-1)  # (batch,)
        return logits

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_advanced_classifier(
    config: TrainingConfig, feature_dimension: int
) -> nn.Module:
    """Build TCN, Transformer, or Residual MLP classifier from TrainingConfig."""
    arch = config.architecture
    hidden = config.hidden_size
    layers = config.num_layers
    dropout = config.effective_dropout

    if arch == "tcn":
        return TcnClassifier(
            feature_dimension=feature_dimension,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout,
        )
    elif arch == "transformer":
        return TransformerEncoderClassifier(
            feature_dimension=feature_dimension,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout,
        )
    elif arch == "residual_mlp":
        return ResidualMlpClassifier(
            feature_dimension=feature_dimension,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown advanced architecture: {arch!r}")
