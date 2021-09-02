
import argparse
from extension import backbones

from ExpSDN.model.ExpSDN import creat_ExpSDN

# from extension.config import get_cfg

models_dict = {"ExpSDN":creat_ExpSDN}

def dataset_specific_config_update(config, dset):
    mconfig = config["arch_params"]
    # Query Encoder
    mconfig["query_enc_emb_idim"] = len(list(dset.wtoi.keys()))
    mconfig["loc_word_emb_vocab_size"] = len(list(dset.wtoi.keys()))
    mconfig["dataset"] = dset.config["dataset"]
    return config

def options():
    pass

def make(cfg: argparse.Namespace):
    backbone = backbones.make(cfg)
    # print(cfg)
    model = models_dict[cfg["arch"]](cfg, backbone = backbone)
    return model
