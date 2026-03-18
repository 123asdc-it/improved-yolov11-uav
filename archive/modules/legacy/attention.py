"""
Attention Modules:
  - EMA: Efficient Multi-Scale Attention (ICASSP 2023) [kept for ablation]
  - CA:  Coordinate Attention (CVPR 2021) [used in final model]

CA is stronger than EMA for small object detection:
- Captures long-range spatial information along H and W separately
- Embeds precise positional information into channel attention
- More parameter-efficient: no matrix multiplication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA(nn.Module):
    """Efficient Multi-Scale Attention (kept for ablation comparison)."""

    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        gc = channels // self.groups
        self.gn = nn.BatchNorm2d(gc)
        self.conv1x1 = nn.Conv2d(gc, gc, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(gc, gc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class CA(nn.Module):
    """Coordinate Attention (CVPR 2021).

    Captures long-range spatial dependencies along H and W axes separately,
    embedding precise positional information into channel attention maps.
    More effective than SE/CBAM/EMA for small object detection.

    Paper: Coordinate Attention for Efficient Mobile Network Design
    """

    def __init__(self, channels, reduction=32):
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        # Encode H and W spatial info separately
        x_h = self.pool_h(x)                          # b, c, h, 1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)     # b, c, 1, w -> b, c, w, 1
        # Concatenate along spatial dim and encode
        y = torch.cat([x_h, x_w], dim=2)             # b, c, h+w, 1
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)                # b, mid, 1, w
        # Generate attention maps
        a_h = self.conv_h(x_h).sigmoid()              # b, c, h, 1
        a_w = self.conv_w(x_w).sigmoid()              # b, c, 1, w
        return x * a_h * a_w
