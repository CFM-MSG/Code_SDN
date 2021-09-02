#some utility function

import argparse
import os
from itertools import repeat
from typing import Any, List, Tuple, Union, Iterable

import psutil
# from torch._six import container_abcs


def identity_fn(x, *args, **kwargs):
    """return first input"""
    return x


def Identity_fn(*args, **kwargs):
    """return an identity function"""
    return identity_fn


def str2num(s: str):
    s.strip()
    try:
        value = int(s)
    except ValueError:
        try:
            value = float(s)
        except ValueError:
            if s == 'True':
                value = True
            elif s == 'False':
                value = False
            elif s == 'None':
                value = None
            else:
                value = s
    return value


def str2bool(v):
    if not isinstance(v, str):
        return bool(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def str2dict(s) -> dict:
    if s is None:
        return {}
    if not isinstance(s, str):
        return s
    s = s.split(',')
    d = {}
    for ss in s:
        ss = ss.strip()
        if ss == '':
            continue
        ss = ss.split('=')
        assert len(ss) == 2
        key = ss[0].strip()
        value = str2num(ss[1])
        d[key] = value
    return d


def str2list(s: str) -> list:
    if not isinstance(s, str):
        return list(s)
    items = []
    s = s.split(',')
    for ss in s:
        ss = ss.strip()
        if ss == '':
            continue
        items.append(str2num(ss))
    return items


def str2tuple(s: str) -> tuple:
    return tuple(str2list(s))


def extend_list(l: list, size: int, value=None):
    if value is None:
        value = l[-1]
    while len(l) < size:
        l.append(value)
    return l[:size]


def path(p: str):
    return os.path.abspath(os.path.expanduser(p))


def getRAMinfo(unit=1024 ** 3):
    mem = psutil.virtual_memory()
    return [mem.total / unit, mem.used / unit, mem.cached / unit, mem.free / unit]
    ## Return RAM information (unit=kb) in a list
    ## Index 0: total RAM
    ## Index 1: used RAM
    ## Index 2: free RAM
    # p = os.popen('free -m')
    # i = 0
    # while 1:
    #     i = i + 1
    #     line = p.readline()
    #     if i == 2:
    #         return line.split()[1:4]


def eval_str(s: str):
    return eval(s)


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def add_bool_option(parser, *name_or_flags: str, default=False, help='', **kwargs: Any):
    parser.add_argument(*name_or_flags, nargs='?', const=not default, default=default, type=str2bool, metavar='B',
                        help=help, **kwargs)


def n_tuple(x, n: int) -> tuple:
    if isinstance(x, (tuple, list, set)):
        assert len(x) == n, f"The length is {len(x)} not {n}"
        return tuple(x)
    return tuple(repeat(x, n))
