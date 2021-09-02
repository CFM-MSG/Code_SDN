

from ExpSDN.loss.SDN_loss import create_SDN_Loss



loss_dict = {"sdn_loss": create_SDN_Loss}

def options():
    pass

def make(cfg):
    loss_type = cfg["loss_type"]
    loss_meters = cfg["loss_params"]
    return loss_dict[loss_type](loss_meters)