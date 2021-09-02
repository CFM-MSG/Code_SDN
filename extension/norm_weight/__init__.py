import argparse

from ..config import get_parser
from ..utils import str2dict, str2list, Identity_fn


class _config:
    w_norm = "None"
    w_norm_cfg = {}
    w_norm_ids = [0, 1, 2]

    norm_methods = {
        "None": Identity_fn,
    }


def options(parser=None):
    if parser is None:
        parser = get_parser()
    group = parser.add_argument_group("Weight Normalization Options")
    group.add_argument("--w-norm", default=_config.w_norm,
                       help="Use which normalization layers? {" + ", ".join(
                           _config.norm_methods.keys()) + "}" + " (default: {})".format(_config.w_norm))
    group.add_argument("--w-norm-cfg", type=str2dict, default=_config.w_norm_cfg, metavar="DICT", help="layers config.")
    group.add_argument("--w-norm-ids", metavar='IDs', type=str2list, default=_config.w_norm_ids,
                       help="Which layers will be apply weight normalization?")
    return group


def make(cfg: argparse.Namespace):
    '''
    weight norm func
    norm weight rather than input or output
    '''
    for key, value in vars(cfg).items():
        if key in _config.__dict__:
            setattr(_config, key, value)
    return ("_" + _config.w_norm) if _config.w_norm != "None" else ""


def WeightNorm(No=1, *args, **kwargs):
    kwargs.update(_config.w_norm_cfg)
    if _config.w_norm == "None" or No not in _config.w_norm_ids:
        return Identity_fn()
    return _config.norm_methods[_config.w_norm](*args, **kwargs)
