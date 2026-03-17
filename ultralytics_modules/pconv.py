"""
PConv (Partial Convolution) Module
Paper: Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks (FasterNet, CVPR 2023)
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, autopad


class PConv(nn.Module):
    """Partial Convolution - only convolves a portion of input channels for efficiency."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, n_div=4):
        super().__init__()
        self.dim_conv = c1 // n_div
        self.dim_untouched = c1 - self.dim_conv
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()
        # pointwise conv to adjust channels
        if c1 != c2:
            self.pw = Conv(c1, c2, 1)
        else:
            self.pw = nn.Identity()

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x = self.act(self.bn(torch.cat([x1, x2], dim=1)))
        return self.pw(x)


class PConv_C3k2(nn.Module):
    """C3k2 block with PConv replacing standard convolutions for lightweight design."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        self.m = nn.ModuleList(PConv(c_, c_, 3) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
