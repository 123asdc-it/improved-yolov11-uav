# Active modules — archived modules in archive/modules/
from .simam import SimAM
from .pconv import PConv, PConv_C3k2
from .nwd import (
    nwd, sa_nwd, nwd_loss, sa_nwd_loss, nwd_nms,
    patch_sa_nwd_loss, patch_sa_nwd_tal, patch_nwd_nms,
    patch_nwd_loss, patch_nwd_tal, patch_all_nwd,
    sa_nwd_reverse, patch_sa_nwd_loss_reverse,  # Exp E: reverse C direction
)

__all__ = [
    "SimAM",
    "PConv", "PConv_C3k2",
    "sa_nwd", "sa_nwd_loss", "nwd_nms",
    "patch_sa_nwd_loss", "patch_sa_nwd_tal", "patch_nwd_nms",
    "nwd", "nwd_loss", "patch_nwd_loss", "patch_nwd_tal", "patch_all_nwd",
    "sa_nwd_reverse", "patch_sa_nwd_loss_reverse",
]
