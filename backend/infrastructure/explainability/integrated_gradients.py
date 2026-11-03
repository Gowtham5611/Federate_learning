"""PyTorch-native Integrated Gradients explainer for recurrent sequence classifiers.

Axiomatic attribution implementation adhering to Sundararajan et al. (2017)
'Axiomatic Attribution for Deep Networks'. Designed specifically for sequence
models with packed padded inputs (pack_padded_sequence).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn


class SequenceIntegratedGradients:
    """Computes path-integrated gradients for sequence models with length masking.

    Attributes:
        model: PyTorch nn.Module operating on (features, lengths).
        target: 'logit' (pre-sigmoid output) or 'prob' (sigmoid probability).
    """

    def __init__(self, model: nn.Module, target: str = "logit") -> None:
        if target not in ("logit", "prob"):
            raise ValueError(f"Target must be 'logit' or 'prob', got {target!r}")
        self.model = model
        self.target = target
        self.model.eval()

    def _forward_target(self, features: Tensor, lengths: Tensor) -> Tensor:
        """Evaluate scalar target per sequence."""
        logits = self.model(features, lengths).view(-1)
        if self.target == "prob":
            return torch.sigmoid(logits)
        return logits

    def attribute(
        self,
        inputs: Tensor,
        lengths: Tensor,
        baseline: Optional[Tensor] = None,
        steps: int = 50,
        batch_step_size: int = 16,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute Integrated Gradients attributions for a batch of sequences.

        Args:
            inputs: (B, T, D) float tensor of input sequences.
            lengths: (B,) int64 tensor of true sequence lengths.
            baseline: (B, T, D) or (1, T, D) float tensor. If None, zero baseline
                is used, masked to zero beyond each sequence's length.
            steps: Number of Riemann integration steps along the straight-line path.
            batch_step_size: Mini-batch size along the interpolation axis to
                bound memory footprint.

        Returns:
            attributions: (B, T, D) float tensor of integrated gradient values.
            diagnostics: Dictionary containing completeness residual, convergence
                checks, runtime, and model outputs.
        """
        device = inputs.device
        batch_size, seq_len, feat_dim = inputs.shape
        t0 = time.time()

        if baseline is None:
            baseline = torch.zeros_like(inputs)
        elif baseline.shape[0] == 1 and batch_size > 1:
            baseline = baseline.expand(batch_size, seq_len, feat_dim).clone()

        # Enforce padding mask on baseline: positions >= length must be zero
        mask = torch.arange(seq_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        baseline = baseline.masked_fill(mask.unsqueeze(-1), 0.0)
        # Ensure inputs also have zeros on padding
        inputs_masked = inputs.masked_fill(mask.unsqueeze(-1), 0.0)

        diff = inputs_masked - baseline  # (B, T, D)

        # Evaluate target at endpoints for completeness verification
        with torch.no_grad():
            f_input = self._forward_target(inputs_masked, lengths)
            f_baseline = self._forward_target(baseline, lengths)
            f_diff = f_input - f_baseline  # (B,)

        # Accumulate gradients using midpoint Riemann integration: alpha = (k - 0.5) / steps
        grad_sum = torch.zeros_like(inputs_masked)

        for step in range(steps):
            alpha = (step + 0.5) / float(steps)
            x_step = (baseline + alpha * diff).detach().requires_grad_(True)

            out_step = self._forward_target(x_step, lengths)
            # Scalar sum for backprop
            grad_outputs = torch.ones_like(out_step)
            grads = torch.autograd.grad(
                outputs=out_step,
                inputs=x_step,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
            )[0]

            # Mask out any spurious gradient in padding positions
            grads = grads.masked_fill(mask.unsqueeze(-1), 0.0)
            grad_sum += grads.detach()

        avg_grads = grad_sum / float(steps)
        attributions = diff * avg_grads  # (B, T, D)

        # Completeness check: sum(IG) vs (F(x) - F(x0))
        attr_sum = attributions.sum(dim=(1, 2))  # (B,)
        abs_residuals = (attr_sum - f_diff).abs()
        denom = f_diff.abs() + 1e-7
        rel_residuals = abs_residuals / denom

        elapsed = time.time() - t0

        diagnostics = {
            "target": self.target,
            "steps": steps,
            "runtime_seconds": elapsed,
            "f_input": f_input.detach().cpu().numpy(),
            "f_baseline": f_baseline.detach().cpu().numpy(),
            "f_diff": f_diff.detach().cpu().numpy(),
            "attr_sum": attr_sum.detach().cpu().numpy(),
            "abs_residuals": abs_residuals.detach().cpu().numpy(),
            "rel_residuals": rel_residuals.detach().cpu().numpy(),
            "max_abs_residual": float(abs_residuals.max().item()),
            "mean_abs_residual": float(abs_residuals.mean().item()),
            "max_rel_residual": float(rel_residuals.max().item()),
            "mean_rel_residual": float(rel_residuals.mean().item()),
        }

        return attributions.detach(), diagnostics

