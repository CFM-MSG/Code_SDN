import argparse

from ..config import get_parser
from ..utils import str2dict, str2list



class _config:
    w_norm = "None"
    w_norm_cfg = {}
    w_norm_ids = [0]

    norm_methods = {
        
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
    pass