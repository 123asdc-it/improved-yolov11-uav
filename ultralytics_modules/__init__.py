from .attention import EMA, CA  # kept for reference / old configs
from .simam import SimAM
from .pconv import PConv, PConv_C3k2
from .bifpn import BiFPN_Concat
from .repvgg import RepVGGBlock, repvgg_model_convert
from .carafe import CARAFE
from .inner_iou import inner_iou, inner_ciou_loss, patch_ultralytics_loss  # deprecated
from .nwd import (
    nwd, sa_nwd, nwd_loss, sa_nwd_loss, nwd_nms,
    patch_sa_nwd_loss, patch_sa_nwd_tal, patch_nwd_nms,
    patch_nwd_loss, patch_nwd_tal, patch_all_nwd,
)

__all__ = [
    # New approach (SimAM + SA-NWD)
    "SimAM",
    "sa_nwd", "sa_nwd_loss", "nwd_nms",
    "patch_sa_nwd_loss", "patch_sa_nwd_tal", "patch_nwd_nms",
    "nwd", "nwd_loss", "patch_nwd_loss", "patch_nwd_tal", "patch_all_nwd",
    # Kept modules
    "PConv", "PConv_C3k2",
    "BiFPN_Concat",
    "RepVGGBlock", "repvgg_model_convert",
    "CARAFE",
    # Legacy (kept for reference)
    "EMA", "CA",
    "inner_iou", "inner_ciou_loss", "patch_ultralytics_loss",
]
