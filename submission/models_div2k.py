"""Factory for the uploaded v1/v2 shared backbone; no cloud-data/loss imports."""
from .cloud_removal_v1.models.vlifnet import build_vlifnet
from .cloud_removal_v1.models.plain_unet import build_plain_unet


def build_model(backbone='snn', dim=24, use_refinement=False):
    if backbone == 'plain':
        if use_refinement:
            raise ValueError('PlainUNet has no optional refinement blocks')
        return build_plain_unet(dim=dim)
    if backbone not in ('snn', 'ann'):
        raise ValueError('backbone must be snn, ann or plain')
    return build_vlifnet(dim=dim, T=4, backend='torch', bn_variant='tdbn',
                         backbone=backbone, use_refinement=use_refinement)
