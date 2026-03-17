"""
RepVGG Block (CVPR 2021)
Paper: RepVGG: Making VGG-style ConvNets Great Again

Train: multi-branch (3x3 + 1x1 + identity)
Infer: single 3x3 conv (reparameterized) -> faster, same accuracy
"""

import numpy as np
import torch
import torch.nn as nn


class RepVGGBlock(nn.Module):
    """RepVGG building block with structural re-parameterization."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        assert kernel_size == 3
        assert padding == 1

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, groups=groups, bias=True)
        else:
            self.rbr_3x3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1,
                          groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0,
                          groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            # Identity branch only when in==out and stride==1
            self.rbr_identity = nn.BatchNorm2d(in_channels) \
                if (out_channels == in_channels and stride == 1) else None

        self.act = nn.SiLU()

    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr_reparam(x))

        out = self.rbr_3x3(x)
        if self.rbr_1x1 is not None:
            out = out + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.act(out)

    def switch_to_deploy(self):
        """Reparameterize all branches into a single conv for fast inference."""
        if self.deploy:
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_channels, self.out_channels, 3, self.stride, 1,
            groups=self.groups, bias=True
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        # Remove training branches
        for attr in ['rbr_3x3', 'rbr_1x1', 'rbr_identity']:
            if hasattr(self, attr):
                delattr(self, attr)
        self.deploy = True

    def _pad_1x1_to_3x3(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            bn = branch[1]
        else:  # BN only (identity)
            assert isinstance(branch, nn.BatchNorm2d)
            kernel = torch.eye(self.in_channels, dtype=branch.weight.dtype,
                               device=branch.weight.device)
            kernel = kernel.view(self.in_channels, self.in_channels, 1, 1)
            bn = branch
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def _get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_bn(self.rbr_3x3)
        k1, b1 = self._fuse_bn(self.rbr_1x1)
        kid, bid = self._fuse_bn(self.rbr_identity)
        return (k3 + self._pad_1x1_to_3x3(k1) + self._pad_1x1_to_3x3(kid),
                b3 + b1 + bid)


def repvgg_model_convert(model, save_path=None):
    """Convert all RepVGGBlock in model to deploy mode."""
    for module in model.modules():
        if isinstance(module, RepVGGBlock):
            module.switch_to_deploy()
    if save_path:
        torch.save(model.state_dict(), save_path)
    return model
