from extension import backbones
# from .simple_backbone import simple_backbone as sb
from .vgg import vgg
backbones_dict = {'vgg': vgg}




def options():
    pass

def make(cfg):
    if "backbone" not in cfg.keys():
        return None
    return backbones_dict[cfg["backbone"]](cfg["backbone_out"], cfg["backbone_arch"])