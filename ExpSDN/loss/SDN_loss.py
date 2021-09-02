import torch
import pdb

import torch.nn as nn



def create_SDN_Loss(loss_args):
    return _SDN_Loss(loss_args)

class _SDN_Loss(torch.nn.Module):
    def __init__(self, mconfig):
        super().__init__()
        self.args = mconfig
        # build criterion
        self.USE_ATTENTION_LOSS = mconfig["USE_ATTENTION_LOSS"]
        self.USE_MID_LOSS = mconfig["USE_MID_LOSS"]
        self.mse = nn.MSELoss()

    def kl_div(self, p, gt, length):
        individual_loss = []
        for i in range(length.size(0)):
            vlength = int(length[i])
            ret = gt[i][:vlength] * torch.log((p[i][:vlength]/gt[i][:vlength]))
            individual_loss.append(-torch.sum(ret))
            if torch.any(torch.isnan(ret)):
                pdb.set_trace()
        individual_loss = torch.stack(individual_loss)
        return torch.mean(individual_loss), individual_loss

    def get_mask_from_sequence_lengths(self, sequence_lengths: torch.Tensor, max_length: int):
        ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
        range_tensor = ones.cumsum(dim=1)
        return (sequence_lengths.unsqueeze(1) >= range_tensor).long()
    
    def complete_loss(self, pred, gt, length):
        individual_loss = []

        for i in range(length.size(0)):
            vlength = int(length[i])

            ret = 0.7*(-gt[i][:vlength] * torch.log(pred[i][:vlength])).sum(dim = -1)/(sum(gt[i][:vlength])+1e-8)
            if torch.any(torch.isnan(ret)):
                pdb.set_trace()
            individual_loss.append(torch.sum(ret))
        individual_loss = torch.stack(individual_loss)
        return torch.mean(individual_loss), individual_loss


    def max_ll(self, pred, loc):
        pred_loss = -(torch.log(pred) * loc).sum(dim = -1)
        return torch.mean(pred_loss), pred_loss

    def compute_tag(self, tag_attw, gt_loc):
        
        ac_loss = (-gt_loc*torch.log(tag_attw+1e-8)).sum(1) / (gt_loc.sum(1)+1e-8)
        ac_loss = (0.5 * ac_loss.mean(0))
        if torch.any(torch.isnan(ac_loss)):
            pdb.set_trace()
        return ac_loss

    def forward(self, output, target):
        pred_start = output["pred_start"]
        pred_end = output["pred_end"]

        
        
        start = target["start"]
        end = target["end"]
        videoFeat_lengths = target["videoFeat_lengths"]
        localiz = target["localiz"]

        if "video_mask" in output.keys():
            videoFeat_lengths = output["video_mask"].sum(-1)
            localiz = output["video_mask"]

        start_loss, individual_start_loss = self.kl_div(pred_start, start, videoFeat_lengths)
        end_loss, individual_end_loss     = self.kl_div(pred_end, end, videoFeat_lengths)


        individual_loss = individual_start_loss + individual_end_loss
        

        loss = {}

        if self.USE_ATTENTION_LOSS:
            pred_attw = output["pred_attw"]
            attw_loss = self.compute_tag(pred_attw, localiz)

            total_loss = start_loss + end_loss + attw_loss
            loss["attw_loss"] = attw_loss

        else:
            total_loss = start_loss + end_loss

        if self.USE_MID_LOSS:
            pred_mid = output["pred_mid"]
            mid_loss, individual_mid_loss = self.complete_loss(pred_mid, localiz, videoFeat_lengths)
            total_loss += mid_loss
            loss["mid_loss"] = mid_loss
            individual_loss += individual_mid_loss

        loss["total_loss"] = total_loss
        loss["start_loss"] = start_loss
        loss["end_loss"] = end_loss
        
        return loss



