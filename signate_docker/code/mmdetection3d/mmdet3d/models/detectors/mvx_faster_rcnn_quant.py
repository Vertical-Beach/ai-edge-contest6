import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from .mvx_two_stage_quant import MVXTwoStageDetector_quant


@DETECTORS.register_module()
class MVXFasterRCNN_quant(MVXTwoStageDetector_quant):
    """Multi-modality VoxelNet using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(MVXFasterRCNN_quant, self).__init__(**kwargs)

