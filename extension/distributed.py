import argparse
import os
import pickle

import torch
import torch.distributed

from extension.config import get_parser
from extension.utils import add_bool_option

def options(parser=None):
    if parser is None:
        parser = get_parser()
    group = parser.add_argument_group("Distributed Training Options")
    group.add_argument("-d", "--distributed", action="store_true", default=False, help="Use distributed training?")
    # help=argparse.SUPPRESS)
    group.add_argument("--local_rank", metavar="N", type=int, default=0, help="GPU id to use.")
    group.add_argument("--dist-backend", metavar="C", default="nccl", choices=["nccl", "gloo", "mpi"],
                       help="The backend type [nccl, gloo, mpi].")
    group.add_argument("--dist-method", metavar="S", default="env://", help="The init_method.")
    add_bool_option(group, "--dist-apex", default=False, help="use apex.parallel.DistributedDataParallel?")
    add_bool_option(group, "--sync-bn", default=False, help="Change the BN to SyncBN?")
    return group

def make(cfg: argparse.Namespace):
    device = torch.device('cuda', cfg.local_rank)
    torch.cuda.set_device(cfg.local_rank)
    # distributed training
    WORLD_SIZE = os.getenv("WORLD_SIZE")
    cfg.distributed = (WORLD_SIZE is not None) and int(WORLD_SIZE) > 0
    if cfg.distributed:
        torch.distributed.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_method)
    return device

# utils function for distributed env
def get_rank():
    #some error occured, ignored temporarily
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()
    # return 0

def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()
    
def is_main_process():
    return get_rank() == 0

def synchronize():
    if get_world_size() > 1:
        torch.distributed.barrier()
    # if get_world_size() == 1:
    #     return
    # rank = torch.distributed.get_rank()
    #
    # def _send_and_wait(r):
    #     if rank == r:
    #         tensor = torch.tensor(0, device="cuda")
    #     else:
    #         tensor = torch.tensor(1, device="cuda")
    #     torch.distributed.broadcast(tensor, r)
    #     while tensor.item() == 1:
    #         time.sleep(1)
    #
    # _send_and_wait(0)
    # _send_and_wait(1) # now sync on the main process


def reduce_tensor(tensor: torch.Tensor, return_mean = True) -> torch.Tensor:
    if get_world_size() == 1:
        return tensor
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    if return_mean:
        rt /= get_world_size()
    return rt