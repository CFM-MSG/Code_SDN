import argparse
import random
import time
import numpy as np
import os

import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import torch.utils.data

from extension.config import get_parser
from extension.logger import get_logger
from extension.utils import add_bool_option
from extension.distributed import get_rank

def options(parser=None):
    if parser is None:
        parser = get_parser()
    group = parser.add_argument_group("Train Options")
    group.add_argument("-n", "--epochs", default=90, type=int, metavar="N", help="The total number of training epochs.")
    # group.add_argument("--start-epoch", default=-1, type=int, metavar="N",
    #                    help="manual epoch number (useful on restarts)")
    group.add_argument("-o", "--output", default="./results", metavar="P",
                       help="The root path to store results (default ./results)")
    add_bool_option(group, "--test", default=False, help="Only test model on test set?")
    add_bool_option(group, "--eval", default=False, help="Only test model on validation set?")
    group.add_argument("--seed", default=-1, type=int, metavar='N', help="manual seed")
    # add_bool_option(group, "--debug", default=False, help="Debug this program?")
    add_bool_option(group, "--fp16", default=False, help="Use mixed-precision to train network")
    group.add_argument("--grad-clip", default=-1, type=float, metavar='V',
                       help="The value of max norm when perform gradient clip (>0)")
    return group

def make(cfg: argparse.Namespace):
    logger = get_logger()
    # logger("==> args: {}".format(cfg))
    # logger("==> Config:")
    # for k, v in sorted(vars(cfg).items()):
    #     logger("{}: {}".format(k, v))
    # guarantee model reproducible
    if not hasattr(cfg, "seed") or cfg.seed < 0:
        cfg.seed = int(time.time())
    else:
        cfg.seed = cfg.seed + get_rank()  # let seed be different(distributed training)
        cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    logger("==> seed: {}".format(cfg.seed))
    logger("==> PyTorch version: {}, cudnn version: {}, CUDA compute capability: {}.{}".format(
        torch.__version__, cudnn.version(), *torch.cuda.get_device_capability()))
    git_version = os.popen("git log --pretty=oneline | head -n 1").readline()[:-1]
    logger("==> git version: {}".format(git_version))
    if cfg.grad_clip > 0:
        logger("==> the max norm of gradient clip is {}".format(cfg.grad_clip))
    return
