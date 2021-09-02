import argparse
import warnings

import torch
import yaml

_parser = None
_config = None


def get_parser(*args, **kwargs):
    '''
    generate a global parser
    return : ArgumentParser instance
    '''
    global _parser
    if _parser is None:
        _parser = argparse.ArgumentParser(*args, **kwargs)
    return _parser


def get_config():
    '''
    return global config
    '''
    return _config


def get_cfg(key, default=None):
    '''
    get an attributes key's value from global config
    return: value of key ->int/float/str/list/dict
    '''
    if not hasattr(_config, key):
        warnings.warn("\033[41mNo such {} item in config, use default {}\033[0m".format(key, default))
    return getattr(_config, key, default)


def set_cfg(key, value):
    '''
    set values in cfg according to keys
    return: target value
    '''
    setattr(_config, key, value)
    return getattr(_config, key)


def _load_from_yaml(filename, cfg, handle_unknown='interrupt'):
    '''
    load cfg from yaml file.
    input: 
        filename:path of yaml file
        cfg: global config
        handle_unknow: action type when get an unknown keys
    '''
    if not filename:
        return cfg
    with open(filename, "r") as f:
        if yaml.__version__ < '5.1':
            yaml_cfg = yaml.load(f)
        else:
            yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        for k, v in yaml_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                if handle_unknown == 'ignore':
                    warnings.warn("Unknown option {k}={v}")
                elif handle_unknown == 'interrupt':
                    raise AttributeError("Unknown option {k}={v}")
                elif handle_unknown == 'add':
                    # warnings.warn("Unknown option {k}={v} found, adding into config...")
                    setattr(cfg, k, v)
                else:
                    setattr(cfg, k, v)
        print("\033[32mLoad config from yaml file: {}\033[0m".format(filename))
    return cfg


def _load_from_pth(filename, cfg=argparse.Namespace()):
    '''
    load data from an checkpoint file
    checkpoint = model + cfg
    use pop to extract cfg and leave model for model-init
    '''
    if bool(filename):
        pth = torch.load(filename, map_location="cpu")
        if isinstance(pth, dict) and "cfg" in pth:
            _cfg = pth.pop("cfg")
            if isinstance(_cfg, argparse.Namespace):
                for k, v in _cfg.__dict__.items():
                    setattr(cfg, k, v)
        from .checkpoint import set_checkpoint
        set_checkpoint(pth)
        print("\033[32mLoad checkpoint from: `{}`\033[0m".format(filename))
    return cfg


def make(args=None, ignore_unknown=False):
    global _config
    parser = get_parser()
    cfg = parser.parse_args(args)
    yaml_file = getattr(cfg, "yaml", None)
    resume_file = getattr(cfg, "resume", None)

    cfg = parser.parse_args([])
    cfg = _load_from_pth(resume_file, cfg)
    cfg = _load_from_yaml(yaml_file, cfg, handle_unknown = 'add')
    _config = parser.parse_args(args, cfg)
    return _config


def options(parser=None):
    group = (parser if parser else get_parser()).add_argument_group(
        "Config Options. Load order <resume> <yaml> <cmd-line>. For the same option, only the last one work.")
    group.add_argument("-c", "--yaml", metavar="P", default="", help="Use YAML file to config.")
    return group
