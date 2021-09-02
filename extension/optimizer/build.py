import argparse

import torch
from torch import mode
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.rmsprop import RMSprop
from torch.optim import Adam, SGD

from extension.config import get_parser
from extension.logger import get_logger
# from extension.optimizer.RAdam import RAdam
# from extension.optimizer.ranger import Ranger
from extension.utils import str2dict, add_bool_option

_methods = {
    "sgd": SGD,
    "adam": Adam,
    # "adamax": Adamax,
    # "RMSprop": RMSprop,
    # 'radam': RAdam,
    # 'adamw': AdamW,
    # 'ranger': Ranger
}


def options(parser=None):
    if parser is None:
        parser = get_parser()
    group = parser.add_argument_group("Optimizer Option:")
    group.add_argument("-oo", "--optimizer", default="sgd", choices=_methods.keys(), metavar='C',
                       help="the optimizer method to train network {" + ", ".join(_methods.keys()) + "}")
    group.add_argument("-oc", "--optimizer-cfg", default={}, type=str2dict, metavar="D",
                       help="The configure for optimizer")
    group.add_argument("-wd", "--weight-decay", default=0, type=float, metavar="V",
                       help="weight decay (default: 0).")
    
    add_bool_option(group, "--no-wd-bias", default=False, help="Set the weight decay on bias and bn to 0")
    group.add_argument("-blr", "--bias_lr_factor", default=1, type=float, metavar="V",
                       help="bias learning rate factor (default: 1).")

def _add_weight_decay(net, l2_value, skip_list=()):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': decay, 'weight_decay': l2_value}, {'params': no_decay, 'weight_decay': 0.}, ]

def make(model: torch.nn.Module, cfg: argparse.Namespace, params=None, **kwargs):
    '''
    model: torch.nn.Module
    cfg: framework cfg
    kwargs: some configuration of optimizer. e.g. {"momentum":0.9, "lr" : 1e-2}. using yaml is adviced.
    '''
    if cfg.optimizer == "sgd":
        kwargs.setdefault("momentum", 0.9)
    if hasattr(cfg, "lr"):
        kwargs["lr"] = cfg.lr
    kwargs.setdefault("weight_decay", cfg.weight_decay)
    for k, v in cfg.optimizer_cfg.items():
        if k not in kwargs:
            kwargs[k] = v
    
    #network parameters
    if params is None:
        if cfg.no_wd_bias:
            params = _add_weight_decay(model, cfg.weight_decay)
        else:
            params = model.parameters()
    logger = get_logger()
    #choose approximate optimizer.
    optimizer = _methods[kwargs.pop('optimizer') if 'optimizer' in kwargs else cfg.optimizer](params, **kwargs)
    logger("==> Optimizer {}".format(optimizer))
    return optimizer
