"""
CARAFE: Content-Aware ReAssembly of FEatures (ICCV 2019)
Paper: CARAFE: Content-Aware ReAssembly of FEatures

Replaces bilinear/nearest upsample in P2 branch with content-aware
reassembly, which is much more effective for small object features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures upsampling module.

    Args:
        channels (int): Input/output channel count.
        scale_factor (int): Upsampling scale factor (default 2).
        k_enc (int): Kernel size for encoder (default 3).
        k_up (int): Reassembly kernel size (default 5).
    """

    def __init__(self, channels, scale_factor=2, k_enc=3, k_up=5):
        super().__init__()
        self.scale = scale_factor
        self.k_up = k_up
        self.k_enc = k_enc

        # Channel compressor
        self.comp = nn.Conv2d(channels, max(channels // 4, 16),
                              kernel_size=1, bias=False)
        self.bn_comp = nn.BatchNorm2d(max(channels // 4, 16))

        # Kernel predictor: predicts k_up^2 * scale^2 reassembly weights
        self.enc = nn.Conv2d(max(channels // 4, 16),
                             k_up * k_up * scale_factor * scale_factor,
                             kernel_size=k_enc, padding=k_enc // 2, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        b, c, h, w = x.shape
        H, W = h * self.scale, w * self.scale

        # Predict reassembly kernels
        mask = self.act(self.bn_comp(self.comp(x)))    # b, c//4, h, w
        mask = self.enc(mask)                           # b, k^2*s^2, h, w
        mask = mask.view(b, self.k_up * self.k_up,
                         self.scale * self.scale, h, w)
        mask = mask.permute(0, 2, 1, 3, 4)             # b, s^2, k^2, h, w
        mask = mask.reshape(b * self.scale * self.scale,
                            self.k_up * self.k_up, h, w)
        mask = F.softmax(mask, dim=1)                  # normalize over k^2

        # Unfold input for content-aware reassembly
        x_unf = F.unfold(x, kernel_size=self.k_up,
                         padding=self.k_up // 2)        # b, c*k^2, h*w
        x_unf = x_unf.view(b, c, self.k_up * self.k_up, h * w)
        x_unf = x_unf.permute(0, 3, 1, 2)              # b, h*w, c, k^2
        x_unf = x_unf.reshape(b * h * w, c, self.k_up * self.k_up)

        # mask: b*s^2, k^2, h, w => b*s^2*h*w, 1, k^2
        mask = mask.permute(0, 2, 3, 1)                 # b*s^2, h, w, k^2
        mask = mask.reshape(b, self.scale * self.scale,
                            h * w, self.k_up * self.k_up)
        mask = mask.permute(0, 2, 1, 3)                 # b, h*w, s^2, k^2
        mask = mask.reshape(b * h * w, self.scale * self.scale,
                            self.k_up * self.k_up)

        # Weighted sum: (b*h*w, s^2, k^2) x (b*h*w, k^2, c) -> (b*h*w, s^2, c)
        out = torch.bmm(mask, x_unf.permute(0, 2, 1))  # b*h*w, s^2, c
        out = out.reshape(b, h * w, self.scale * self.scale, c)
        out = out.permute(0, 3, 1, 2)                   # b, c, h*w, s^2
        out = out.reshape(b, c, h, w, self.scale, self.scale)
        out = out.permute(0, 1, 2, 4, 3, 5)             # b, c, h, s, w, s
        out = out.reshape(b, c, H, W)
        return out
