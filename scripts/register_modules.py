"""
Register custom modules with ultralytics.
Import this module BEFORE creating any YOLO model to enable custom layers.

Modules registered:
  - SimAM:       Parameter-Free Attention (ICML 2021) [primary]
  - PConv:       Partial Convolution (FasterNet, CVPR 2023)
  - PConv_C3k2:  Lightweight C3k2 with PConv
  - BiFPN_Concat: Weighted feature fusion (EfficientDet, CVPR 2020)
  - RepVGGBlock: Structural re-parameterization (CVPR 2021)
  - CARAFE:      Content-Aware ReAssembly upsampling (ICCV 2019)
  - EMA:         Efficient Multi-Scale Attention (ICASSP 2023) [legacy]
  - CA:          Coordinate Attention (CVPR 2021) [legacy]

Loss patch (see ultralytics_modules/nwd.py):
  - patch_all_nwd(): Replace CIoU with NWD loss + NWD-TAL label assignment
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ultralytics_modules import (
    SimAM, PConv, PConv_C3k2, BiFPN_Concat,
    RepVGGBlock, CARAFE
)
import ultralytics.nn.tasks as tasks

# Register all custom modules into tasks namespace
for name, cls in [
    ("SimAM", SimAM),
    ("PConv", PConv), ("PConv_C3k2", PConv_C3k2),
    ("BiFPN_Concat", BiFPN_Concat),
    ("RepVGGBlock", RepVGGBlock),
    ("CARAFE", CARAFE),
]:
    setattr(tasks, name, cls)

# Patch parse_model
_original_parse_model = tasks.parse_model


def _patched_parse_model(d, ch, verbose=True):
    """Patched parse_model supporting CA, EMA, PConv_C3k2, BiFPN_Concat, RepVGGBlock, CARAFE."""
    import ast
    import contextlib
    import torch
    from ultralytics.nn.modules import (
        AIFI, C1, C2, C2PSA, C3, C3TR, ELAN1, OBB, PSA, SPP, SPPELAN, SPPF,
        A2C2f, AConv, ADown, Bottleneck, BottleneckCSP, C2f, C2fAttn, C2fCIB,
        C2fPSA, C3Ghost, C3k2, C3x, CBFuse, CBLinear, Classify, Concat, Conv,
        Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d, Focus,
        GhostBottleneck, GhostConv, HGBlock, HGStem, ImagePoolingAttn,
        RepC3, RepConv, RepNCSPELAN4, SCDown,
    )
    from ultralytics.utils import LOGGER
    try:
        from ultralytics.utils.torch_utils import make_divisible
    except ImportError:
        from ultralytics.utils.ops import make_divisible
    try:
        from ultralytics.nn.modules import (
            OBB26, Pose, Pose26, Segment, Segment26, WorldDetect, v10Detect,
            RTDETRDecoder, ResNetLayer, Index, TorchVision, RepVGGDW, LRPCHead,
            YOLOEDetect, YOLOESegment, YOLOESegment26,
        )
    except ImportError:
        pass
    from ultralytics.utils import colorstr

    legacy = True
    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]

    base_modules = frozenset({
        Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck,
        SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP,
        C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN,
        C2fAttn, C3, C3TR, C3Ghost, torch.nn.ConvTranspose2d,
        DWConvTranspose2d, C3x, RepC3, PSA, SCDown, C2fCIB, A2C2f,
        PConv_C3k2,
    })
    repeat_modules = frozenset({
        BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost,
        C3x, RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f,
        PConv_C3k2,
    })

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(tasks, m, None) or eval(m)
        )
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n

        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m is C3k2:
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.2))
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is BiFPN_Concat:
            c2 = sum(ch[x] for x in f)
            args = [1, len(f)]  # dimension, num_inputs
        elif m in frozenset({Detect}):
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m in {Detect}:
                m.legacy = legacy
        elif m in {SimAM}:  # pass-through attention: same channels
            c2 = ch[f]
            args = [c2]
        elif m is RepVGGBlock:  # [out_channels, stride]
            c1 = ch[f]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            stride = args[1] if len(args) > 1 else 1
            args = [c1, c2, 3, stride]
        elif m is CARAFE:  # [channels, scale_factor]
            c2 = ch[f]
            args = [c2, *args]  # prepend channels
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


tasks.parse_model = _patched_parse_model
print("\u2713 Custom modules registered: SimAM, PConv, PConv_C3k2, BiFPN_Concat, RepVGGBlock, CARAFE")
