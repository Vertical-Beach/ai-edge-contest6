import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from .mvx_two_stage_qat import MVXTwoStageDetector_qat


@DETECTORS.register_module()
class MVXFasterRCNN_qat(MVXTwoStageDetector_qat):
    """Multi-modality VoxelNet using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(MVXFasterRCNN_qat, self).__init__(**kwargs)

