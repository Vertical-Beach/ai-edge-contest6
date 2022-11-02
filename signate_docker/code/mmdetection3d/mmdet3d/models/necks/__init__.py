from mmdet.models.necks.fpn import FPN
from .second_fpn import SECONDFPN
from .second_fpn_quant import SECONDFPN_quant

__all__ = ['FPN', 'SECONDFPN',
           'SECONDFPN_quant']
