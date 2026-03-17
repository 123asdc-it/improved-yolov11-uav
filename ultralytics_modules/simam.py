"""
SimAM: A Simple, Parameter-Free Attention Module (ICML 2021)

Key advantage for small object detection:
- ZERO additional parameters → cannot overfit on small datasets
- Based on neuroscience energy theory: enhances neurons with
  discriminative activation patterns (i.e., small object features)
- Drop-in replacement for EMA/CA/SE/CBAM

Paper: SimAM: A Simple, Parameter-Free Attention Module for CNNs
"""

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """Simple, Parameter-Free Attention Module.

    Derives 3D (channel + spatial) attention weights from an energy function
    based on neuroscience theory. No learnable parameters.

    Args:
        e_lambda (float): Regularization constant to avoid division by zero.
    """

    def __init__(self, channels=None, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        # channels arg accepted for compatibility with parse_model but unused

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        # Compute per-neuron energy: how different each activation is from its spatial mean
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # Attention weight: inverse of energy (lower energy = more important)
        y = x_minus_mu_sq / (4 * (x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)
