import argparse

import torch.nn as nn

from ..config import get_parser
from ..utils import str2dict

def _BatchNorm(num_features, dim=4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, *args, **kwargs):
    return (nn.BatchNorm2d if dim == 4 else nn.BatchNorm1d)(
        num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

def _GroupNorm(num_features, num_groups=32, eps=1e-5, affine=True, *args, **kwargs):
    return nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine)


def _LayerNorm(normalized_shape, eps=1e-5, affine=True, *args, **kwargs):
    return nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)


def _InstanceNorm(
        num_features, dim=4, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, *args, **kwargs):
    return (nn.InstanceNorm2d if dim == 4 else nn.InstanceNorm1d)(
        num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)


class _config:
    norm = "BN"
    norm_cfg = {}
    norm_methods = {
        "BN": _BatchNorm,
        "GN": _GroupNorm,
        "LN": _LayerNorm,
        "IN": _InstanceNorm
    }


def Norm(*args, **kwargs):
    kwargs.update(_config.norm_cfg)
    if _config.norm == "None":
        return None
    return _config.norm_methods[_config.norm](*args, **kwargs)


def BatchNorm2d(*args, **kwargs):
    """for instead of nn.BatchNorm2d"""
    return Norm(*args, **kwargs, dim=4)


def BatchNorm1d(*args, **kwargs):
    """for instead of nn.BatchNorm1d"""
    return Norm(*args, **kwargs, dim=2)

def options(parser=None):
    if parser is None:
        parser = get_parser()
    group = parser.add_argument_group("Normalization Options")
    group.add_argument("--norm", default=_config.norm, metavar='C',
                       help="Use which normalization layers? {" + ", ".join(
                           _config.norm_methods.keys()) + "}" + " (default: {})".format(_config.norm))
    group.add_argument("--norm-cfg", type=str2dict, default={}, metavar="D", help="layers config.")
    return group

def make(cfg: argparse.Namespace):
    '''
    using Norm instead of Pytorch's kinds of Norm func
    which norms func will be used is depending on params in YAML or commands
    '''
    for key, value in vars(cfg).items():
        if key in _config.__dict__:
            setattr(_config, key, value)
    return ("_" + _config.norm) if _config.norm != "BN" else ""