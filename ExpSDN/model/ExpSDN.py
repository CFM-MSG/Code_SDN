from torch.nn.modules.linear import Linear
import torch
import torch.nn.functional as F
from torch import nn

import pdb

def creat_ExpSDN(cfg, backbone = None):
    return _ExpSDN(cfg['arch_params'])

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        # x = F.dropout(x, p=self.dropout_rate)
        return x

class GlobalPool1D(nn.Module):
    def __init__(self, idim, odim, ksize = 15, stride = 1, sample_num = -1):
        super(GlobalPool1D, self).__init__() # Must call super __init__()
        
        self.ksize = ksize
        self.stride = stride
        self.pool_layer = nn.MaxPool1d(ksize, stride=stride)
        self.Embedding = VideoEmbedding(idim, odim, sample_num=sample_num)

    def get_global_mask(self, video_mask, ksize = 8, stride = 8):
        new_mask = F.avg_pool1d(video_mask.unsqueeze(1).float(), kernel_size = ksize, stride = stride)[:,0,:]
        new_mask[new_mask < 0.2] = 0
        new_mask[new_mask > 0] = 1
        return new_mask

    def forward(self, feats, video_mask):
        mask = self.get_global_mask(video_mask, ksize=self.ksize, stride= self.stride)
        span = self.pool_layer(feats.permute(0, 2, 1)).permute(0, 2, 1)
        
        out = self.Embedding(span, mask)
        return out, mask

class AttentiveQuery(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_gru = nn.GRU(input_size   = 300,
                                        hidden_size  = 256,
                                        num_layers   = 2,
                                        bias         = True,
                                        dropout      = 0.5,
                                        bidirectional= True,
                                        batch_first = True)

    def forward(self, tokens, tokens_lengths):
        
        packed_tokens = nn.utils.rnn.pack_padded_sequence(tokens, tokens_lengths.data.tolist(), batch_first=True, enforce_sorted=False)
        word_level, _  = self.query_gru(packed_tokens)
        word_level= nn.utils.rnn.pad_packed_sequence(word_level, batch_first=True)[0]

        H = word_level.shape[-1]
        sentence_level = torch.cat((word_level[:,-1,:H//2], word_level[:,0,H//2:]), dim = -1)
        return sentence_level, word_level

class Semantic_Wise_Attention(nn.Module):
    def __init__(self, frame_dim, word_dim, hidden_dim, out_dim):
        super(Semantic_Wise_Attention, self).__init__()

        self.fw_attn_frame_enc = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim, bias=True),
            nn.ReLU()
            )

        self.fw_attn_word_enc = nn.Sequential(
            nn.Linear(word_dim, hidden_dim, bias=True),
            nn.ReLU()
            )
        
        self.fw_attn_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU()
        )

        self.out_head = Linear(hidden_dim, out_dim, bias=False)


    def get_mask_from_sequence_lengths(self, sequence_lengths: torch.Tensor, max_length: int):
        ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
        range_tensor = ones.cumsum(dim=1)
        return (sequence_lengths.unsqueeze(1) >= range_tensor).long()

    def forward(self, frame_feats, word_feats, frame_feats_length, word_feats_length):
        '''
        frame_feats: [B, T, 1024]
        word_feats: [B, L, 512]
        frame_feats_length: [B]
        word_feats_length: [B]
        '''

        v = self.fw_attn_frame_enc(frame_feats)
        h = self.fw_attn_word_enc(word_feats)
        r = self.fw_attn_weight(torch.tanh(v.unsqueeze(2) + h.unsqueeze(1))) #[B, T, L, dim]

        v_mask = self.get_mask_from_sequence_lengths(frame_feats_length, frame_feats.shape[1]).unsqueeze(dim=2).unsqueeze(dim=-1).repeat(1, 1, 1, v.shape[-1]) #[B, T, 1, dim]
        h_mask = self.get_mask_from_sequence_lengths(word_feats_length, word_feats.shape[1]).unsqueeze(dim=1).unsqueeze(dim=-1).repeat(1, 1, 1, v.shape[-1]) #[B, 1, L, dim]

        
        r = r.masked_fill(v_mask.float().eq(0), -1e+32 if r.dtype == torch.float32 else -1e+16)
        r = r.masked_fill(h_mask.float().eq(0), -1e+32 if r.dtype == torch.float32 else -1e+16)
        h2v = torch.nn.functional.softmax(r, dim = -2)

        h2v = h2v.masked_fill(h_mask.float().eq(0), 0)
        h2v = h2v.masked_fill(v_mask.float().eq(0), 0) #[B, T, L, dim]

        frame_wise_word = (h2v.permute(0, 3, 1, 2) @ word_feats.unsqueeze(dim = 1).permute(0, 3, 2, 1)).squeeze().permute(0, 2, 1)
        out = self.out_head(frame_wise_word)
        return out, h2v

class VideoEmbedding(nn.Module):
    def __init__(self, v_idim, v_odim, sample_num = -1):
        super(VideoEmbedding, self).__init__() 

        self.sample_num = 1024 if sample_num < 0 else sample_num

        self.vid_emb_fn = nn.Sequential(*[
            nn.Linear(v_idim, v_odim),
            nn.ReLU(),
            nn.Dropout(0.5)
        ])

    def forward(self, video_feats, video_masks):
        
        video_emb = self.vid_emb_fn(video_feats) * video_masks.unsqueeze(-1)
        
        return video_emb

class PositionEmbedding(nn.Module):
    def __init__(self, v_idim, sample_num = -1):
        super(PositionEmbedding, self).__init__() # Must call super __init__()

        self.sample_num = 1024 if sample_num < 0 else sample_num

        p_idim = 1024 if sample_num < 0 else sample_num
        p_odim = v_idim
        self.pos_emb_fn = nn.Sequential(*[
            nn.Embedding(p_idim, p_odim),
            nn.ReLU(),
            nn.Dropout(0.5)
        ])

    def forward(self, video_feats, video_masks):


        pos = torch.linspace(0, self.sample_num-1, video_feats.size(1)).type_as(video_masks).unsqueeze(0).long()

        pos_emb = self.pos_emb_fn(pos)
        B, nseg, pdim = pos_emb.size()
        pos_feats = (pos_emb.expand(B, nseg, pdim) * video_masks.unsqueeze(2).float())
        video_feats += pos_feats


        return video_feats

class HadamardProduct(nn.Module):
    def __init__(self, idim_1, idim_2, hdim):
        super(HadamardProduct, self).__init__() # Must call super __init__()

        self.fc_1 = nn.Linear(idim_1, hdim)
        self.fc_2 = nn.Linear(idim_2, hdim)
        self.fc_3 = nn.Linear(hdim, hdim)

    def forward(self, x1, x2):
        """
        Args:
            inp1: [B,idim_1] or [B,L,idim_1]
            inp2: [B,idim_2] or [B,L,idim_2]
        """
        # x1, x2 = inp[0], inp[1]
        return torch.relu(self.fc_3(torch.relu(self.fc_1(x1)) * torch.relu(self.fc_2(x2))))

class ResBlock1D(nn.Module):
    def __init__(self, idim, odim, ksize = 15, num_res_blocks = 5, downsample = True):
        super(ResBlock1D, self).__init__() # Must call super __init__()

        self.nblocks = num_res_blocks
        self.do_downsample = downsample

        # set layers
        if self.do_downsample:
            self.downsample = nn.Sequential(
                nn.Conv1d(idim, odim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(odim),
            )
        self.blocks = nn.ModuleList()
        for i in range(self.nblocks):
            cur_block = self.basic_block(idim, odim, ksize)
            self.blocks.append(cur_block)
            if (i == 0) and self.do_downsample:
                idim = odim

    def basic_block(self, idim, odim, ksize=3):
        layers = []
        # 1st conv
        p = ksize // 2
        layers.append(nn.Conv1d(idim, odim, ksize, 1, p, bias=False))
        layers.append(nn.BatchNorm1d(odim))
        layers.append(nn.ReLU(inplace=True))
        # 2nd conv
        layers.append(nn.Conv1d(odim, odim, ksize, 1, p, bias=False))
        layers.append(nn.BatchNorm1d(odim))

        return nn.Sequential(*layers)

    def forward(self, inp):
        """
        Args:
            inp: [B, idim, H, w]
        Returns:
            answer_label : [B, odim, H, w]
        """
        residual = inp.permute(0, 2, 1)

        for i in range(self.nblocks):
            out = self.blocks[i](residual)
            if (i == 0) and self.do_downsample:
                residual = self.downsample(residual)
            out += residual
            out = F.relu(out) # w/o is sometimes better
            residual = out

        return out.permute(0, 2, 1)

class MultiHeadAttention(nn.Module):
    def __init__(self, idim, odim, nhead = 1, use_bias = True):
        super(MultiHeadAttention, self).__init__()

        # dims
        self.idim = idim
        self.odim = odim
        self.nheads = nhead

        # options
        self.use_bias = use_bias

        # layers
        self.c_lin = nn.Linear(self.idim, self.odim*2, bias=self.use_bias)
        self.v_lin = nn.Linear(self.idim, self.odim, bias=self.use_bias)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0)
    def forward(self, m_feats, mask):
        """
        apply muti-head attention
        Inputs:
            m_feats: multimodal features
            mask: mask for features
        Outputs:
            updated_m: updated multimodal features
        """

        mask = mask.float()
        B, nseg = mask.size()
        # key, query, value
        m_k = self.v_lin(self.drop(m_feats)) # [B,num_feats,*]
        m_trans = self.c_lin(self.drop(m_feats))  # [B,num_feats,2*]
        m_q, m_v = torch.split(m_trans, m_trans.size(2) // 2, dim=2)

        new_mq = m_q
        new_mk = m_k

        # applying multi-head attention
        w_list = []
        mk_set = torch.split(new_mk, new_mk.size(2) // self.nheads, dim=2)
        mq_set = torch.split(new_mq, new_mq.size(2) // self.nheads, dim=2)
        mv_set = torch.split(m_v, m_v.size(2) // self.nheads, dim=2)

        for i in range(self.nheads):
            mk_slice, mq_slice, mv_slice = mk_set[i], mq_set[i], mv_set[i] # [B, nseg, *]

            # compute relation matrix; [B,nseg,nseg]
            m2m = mk_slice @ mq_slice.transpose(1,2) / ((self.odim // self.nheads) ** 0.5)

            #be aware of the mask is only valid when the video is short than n_feats
            m2m = m2m.masked_fill(mask.unsqueeze(1).eq(0), -1e+9 if m2m.dtype == torch.float32 else -1e+4) # [B,nseg,nseg]
            m2m_w = F.softmax(m2m, dim=2) # [B,nseg,nseg]
            w_list.append(m2m_w)

            # compute relation vector for each segment
            r = m2m_w @ mv_slice if (i==0) else torch.cat((r, m2m_w @ mv_slice), dim=2) # [B, nseg, dim]
        # print(m_feats.shape, r.shape)
        updated_m = self.drop(m_feats + r)
        return updated_m, torch.stack(w_list, dim=1)

class AttwNetHead(nn.Module):
    def __init__(self, idim, hdim, odim):
        super().__init__()
        self.mlp_attn = nn.Linear(idim, 1, bias=False)
        self.mlp_out = nn.Linear(idim, odim, bias=False)


    def masked_softmax(self, vector: torch.Tensor, mask: torch.Tensor, dim: int = -1, memory_efficient: bool = False, mask_fill_value: float = -1e32):
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                result = torch.nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector, dim=dim)

        return result + 1e-13

    def mask_softmax(self, feat, mask, dim = -1):
        return self.masked_softmax(feat, mask, memory_efficient=True, dim=dim)

    def get_mask_from_sequence_lengths(self, sequence_lengths: torch.Tensor, max_length: int):
        ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
        range_tensor = ones.cumsum(dim=1)
        return (sequence_lengths.unsqueeze(1) >= range_tensor).long()

    def forward(self, mfeats, mask):
        
        logits = self.mlp_attn(mfeats)
        # logits = F.dropout(logits, p=0.5)
        attw = self.mask_softmax(logits, mask.unsqueeze(-1).repeat(1, 1, logits.shape[-1]), dim = 1)
        attn_feats = mfeats * attw
        res = self.mlp_out(attn_feats)
        return res, attw.squeeze()

class ContextEncoder(nn.Module):
    def __init__(self, cfg, ksize=15, num_resblock=5):
        super().__init__()

        self.cfg = cfg

        self.video_modeling = ResBlock1D(512, 512, ksize=ksize, num_res_blocks=num_resblock, downsample=False)
        self.video_context_modeling = MultiHeadAttention(512, 512, nhead=8)

        self.attn_word_level = Semantic_Wise_Attention(512, 512, 512, 512)

        self.local_query_video_fusion = HadamardProduct(512, 512, 512)
        
        self.mfeat_modeling = ResBlock1D(512, 512, ksize=ksize, num_res_blocks=num_resblock, downsample=False)
        self.mfeat_context_modeling = MultiHeadAttention(512, 512, nhead=8)


    def get_mask_from_sequence_lengths(self, sequence_lengths: torch.Tensor, max_length: int):
        ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
        range_tensor = ones.cumsum(dim=1)
        return (sequence_lengths.unsqueeze(1) >= range_tensor).long()

    def forward(self, videoFeat, word_level, videoFeat_lengths, tokens_lengths):
        # assert mask is not None
        video_mask = self.get_mask_from_sequence_lengths(videoFeat_lengths, int(videoFeat.shape[1]))

        video_feats = videoFeat
        
        local_feats = self.video_modeling(video_feats)
        video_context, _ = self.video_context_modeling(local_feats, video_mask)

        frame_wise_word, _ = self.attn_word_level(video_context, word_level, videoFeat_lengths, tokens_lengths)
        mfeats = self.local_query_video_fusion(video_feats, frame_wise_word)

        local_mfeats = self.mfeat_modeling(mfeats)
        mfeats, _ = self.mfeat_context_modeling(local_mfeats, video_mask)

        return mfeats

class MutiLevelEnhance(nn.Module):
    def __init__(self, idim, odim, nhead = 1, use_bias = True):
        super(MutiLevelEnhance, self).__init__()

        # dims
        self.idim = idim
        self.odim = odim
        self.nheads = nhead

        # options
        self.use_bias = use_bias

        # layers
        self.c_lin = nn.Linear(self.idim, self.odim*2, bias=self.use_bias)
        self.v_lin = nn.Linear(self.idim, self.odim, bias=self.use_bias)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0)

        self.out_lin = nn.Linear(2*self.odim, self.odim, bias=False)
    def forward(self, local_feats, global_feats, local_mask, global_mask):

        local_mask = local_mask.float()
        global_mask = global_mask.float()

        # key, query, value
        m_k = self.v_lin(self.drop(local_feats)) # [B,num_seg,*]
        m_trans = self.c_lin(self.drop(global_feats))  # [B,nseg,2*]
        m_q, m_v = torch.split(m_trans, m_trans.size(2) // 2, dim=2)

        new_mq = m_q
        new_mk = m_k

        # applying multi-head attention
        w_list = []
        mk_set = torch.split(new_mk, new_mk.size(2) // self.nheads, dim=2)
        mq_set = torch.split(new_mq, new_mq.size(2) // self.nheads, dim=2)
        mv_set = torch.split(m_v, m_v.size(2) // self.nheads, dim=2)

        for i in range(self.nheads):
            mk_slice, mq_slice, mv_slice = mk_set[i], mq_set[i], mv_set[i] # [B, nseg, *]

            m2m = mk_slice @ mq_slice.transpose(1,2) / ((self.odim // self.nheads) ** 0.5)

            #be aware of the mask is only valid when the video is short than n_feats
            m2m = m2m.masked_fill(global_mask.unsqueeze(1).eq(0), -1e+9 if m2m.dtype == torch.float32 else -1e+4) # [B,nseg,nseg]
            m2m_w = F.softmax(m2m, dim=2) # [B,nseg,nseg]
            w_list.append(m2m_w)

            # compute relation vector for each segment
            r = m2m_w @ mv_slice if (i==0) else torch.cat((r, m2m_w @ mv_slice), dim=2) # [B, nseg, dim]

        updated_m = torch.cat((local_feats, r), dim=-1)
        updated_m = self.out_lin(updated_m)
        return updated_m, torch.stack(w_list, dim=1)

class _ExpSDN(nn.Module):
    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        self.query_proj = AttentiveQuery()
        self.scene_proj = GlobalPool1D(self.cfg["feature_input_dim"], 512, self.cfg["decoupling_scene_width"], self.cfg["decoupling_scene_step"], self.cfg["feature_sample_num"])
        self.action_proj = GlobalPool1D(self.cfg["feature_input_dim"], 512, self.cfg["decoupling_action_width"], self.cfg["decoupling_action_step"], self.cfg["feature_sample_num"])
        self.event_proj = GlobalPool1D(self.cfg["feature_input_dim"], 512, self.cfg["decoupling_event_width"], self.cfg["decoupling_event_step"], self.cfg["feature_sample_num"])

        self.pos_emb = PositionEmbedding(512, self.cfg["feature_sample_num"])

        self.Encoder = nn.ModuleList([
            ContextEncoder(cfg, ksize=self.cfg["modeling_scene_width"], num_resblock=self.cfg["modeling_scene_depth"]),
            ContextEncoder(cfg, ksize=self.cfg["modeling_action_width"], num_resblock=self.cfg["modeling_action_depth"]),
            ContextEncoder(cfg, ksize=self.cfg["modeling_event_width"], num_resblock=self.cfg["modeling_event_depth"])
        ]
        )

        self.aggr = nn.ModuleList(MutiLevelEnhance(512, 512) for i in range(2))
        
        self.head = AttwNetHead(512, 256, 512)
        self.mlp_s = MLP(512, 256, 1, 3)
        self.mlp_m = MLP(512, 256, 1, 3)
        self.mlp_e = MLP(512, 256, 1, 3)

    def get_mask_from_sequence_lengths(self, sequence_lengths: torch.Tensor, max_length: int):
        ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
        range_tensor = ones.cumsum(dim=1)
        return (sequence_lengths.unsqueeze(1) >= range_tensor).long()

    def masked_softmax(self, vector: torch.Tensor, mask: torch.Tensor, dim: int = -1, memory_efficient: bool = False, mask_fill_value: float = -1e32):
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside the mask, we zero these out.
                result = torch.nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector, dim=dim)

        return result + 1e-13

    def mask_softmax(self, feat, mask, dim = -1):
        return self.masked_softmax(feat, mask, memory_efficient=True, dim=dim)

    def forward(self, net_inps):

        # assert mask is not None
        videoFeat = net_inps["videoFeat"]
        videoFeat_lengths = net_inps["videoFeat_lengths"]
        tokens = net_inps['tokens']
        tokens_lengths = net_inps['tokens_lengths']
        video_mask = self.get_mask_from_sequence_lengths(videoFeat_lengths, int(videoFeat.shape[1]))
        # tokens_mask = self.get_mask_from_sequence_lengths(tokens_lengths, int(tokens.shape[1]))

        #SDM
        scene_feat, scene_mask = self.scene_proj(videoFeat, video_mask)
        action_feat, action_mask = self.action_proj(videoFeat, video_mask)
        event_feat, event_mask = self.event_proj(videoFeat, video_mask)
        _, word_level = self.query_proj(tokens, tokens_lengths)

        feats = [scene_feat, action_feat, event_feat]
        masks = [scene_mask, action_mask, event_mask]
        contexts = []

        #SMB
        for i, layer in enumerate(self.Encoder):
            context = layer(self.pos_emb(feats[i], masks[i]), word_level, masks[i].sum(dim = -1), tokens_lengths)
            contexts.append(context)
        
        #Cross-semantic Aggregating
        if len(contexts) > 1:
            action_event_aggr, _ = self.aggr[1](contexts[1], contexts[2], masks[1], masks[2])
            aggr_context, _ = self.aggr[0](contexts[0], action_event_aggr, masks[0], masks[1])
        else:
            aggr_context = contexts[0]

        mfeats, pred_attw = self.head(aggr_context, video_mask)
        if self.cfg['no_attw_regression']:
            mfeats = aggr_context

        logits_s = self.mlp_s(mfeats).squeeze()
        logits_m = self.mlp_m(mfeats).squeeze()
        logits_e = self.mlp_e(mfeats).squeeze()

        pre_start = self.mask_softmax(logits_s, video_mask, dim = 1)
        pre_mid = self.mask_softmax(logits_m, video_mask, dim = 1)
        pre_end = self.mask_softmax(logits_e, video_mask, dim = 1)

        out = {"pred_start": pre_start, "pred_mid": pre_mid, "pred_end": pre_end, "pred_attw": pred_attw}

        return out
