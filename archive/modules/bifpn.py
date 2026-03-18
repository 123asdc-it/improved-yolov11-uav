"""
BiFPN (Bi-directional Feature Pyramid Network) Weighted Concatenation
Paper: EfficientDet: Scalable and Efficient Object Detection (CVPR 2020)

Fix: weights must be registered in __init__ to participate in optimizer.
Use num_inputs parameter to pre-allocate weights.
"""

import torch
import torch.nn as nn


class BiFPN_Concat(nn.Module):
    """BiFPN-style weighted concatenation with fast normalized fusion.

    Drop-in replacement for Concat. Learnable per-input weights are
    registered in __init__ so the optimizer can update them.
    """

    def __init__(self, dimension=1, num_inputs=2):
        super().__init__()
        self.d = dimension
        self.epsilon = 1e-4
        # ★ Register weights in __init__ so optimizer sees them
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        w = torch.relu(self.w)
        w_norm = w / (w.sum() + self.epsilon)
        return torch.cat([w_norm[i] * x[i] for i in range(len(x))], self.d)
