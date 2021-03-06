
import torch
import numpy as np
import torch.nn as nn


def feed_forward_rnn(rnn, embedded_sequence_batch, lengths=None, hidden_tuple=None):
    """
    Recursive function to encapsulate RNN calls.
    :param rnn:
    :param embedded_sequence_batch:
    :param lengths:
    :param hidden_tuple:
    :return:
    """
    if lengths is not None:
        rnn_input, indices_unsort = pack_rnn_input(embedded_sequence_batch, lengths)
        rnn_output, hidden_tuple = rnn(rnn_input, hidden_tuple)
        output = unpack_rnn_output(rnn_output, indices_unsort)
    else:
        output, hidden_tuple = rnn(embedded_sequence_batch, hidden_tuple)

    return output, hidden_tuple

def pad_spatial_sequence(spatial_sequence, batch_first=True):
    
    max_temporal = 0
    max_spatial = 0
    
    length_spatial_feat = []

    for temp in spatial_sequence:
        
        number_frames = len(temp)
        if max_temporal < number_frames:
            max_temporal = number_frames

        number_spatial_feat = []
        for obj in temp:
            aux_num_obj = len(obj)
            if max_spatial < aux_num_obj:
                max_spatial = aux_num_obj
            number_spatial_feat.append(aux_num_obj)
            
        length_spatial_feat.append(number_spatial_feat)
    
    outputSpatial = np.zeros((len(spatial_sequence), max_temporal, max_spatial, 2048), dtype=np.float32)
    outputSpatialLength = np.zeros((len(spatial_sequence), max_temporal), dtype=np.float32)#, max_spatial))
    # print(outputSpatial.shape)

    for i in range(len(spatial_sequence)):
        number_frames = len(spatial_sequence[i])
        for j in range(number_frames):
            # print(i,j, np.array(spatial_sequence[i][j]).shape)
            if len(spatial_sequence[i][j]) == 0:
                continue
            outputSpatial[i, j,:length_spatial_feat[i][j],:] = spatial_sequence[i][j]
            # outputSpatialLength[i,j,:length_spatial_feat[i][j]] = np.ones(length_spatial_feat[i][j])
            outputSpatialLength[i,j] = length_spatial_feat[i][j]
        
    return torch.from_numpy(outputSpatial), torch.from_numpy(outputSpatialLength)


def pad_sequence(sequence, batch_first=True, return_sorted = False, instant_padding = False, padding_num = 128):

    lengths = []
    targets = []
    for idx, s in enumerate(sequence):
        s_len = s.shape[0]

        if instant_padding:
            if s.shape[0] > padding_num:
                idx = np.linspace(0, s.shape[0]-1, num=padding_num, dtype = int)
                s = s[idx]
                lengths.append(padding_num)
            else:
                if len(s.size()) == 2:
                    s = np.pad(s, ((0, padding_num - s.shape[0]), (0, 0)))
                elif len(s.size()) == 1:
                    s = np.pad(s, (0, padding_num - s.shape[0]))
                s = torch.from_numpy(s)
                lengths.append(s_len)
        else:
            lengths.append(s_len)
        targets.append(s)
    lengths = np.array(lengths, dtype=np.float32)
    lengths = torch.from_numpy(lengths)
    padded = nn.utils.rnn.pad_sequence(targets, batch_first=batch_first)
    if return_sorted:
        sorted_lengths, indices = torch.sort(lengths, descending=True)
        sorted_pad = padded[indices]
        return sorted_pad, sorted_lengths
    return padded, lengths

def pack_rnn_input(embedded_sequence_batch, sequence_lengths):
    '''
    :param embedded_sequence_batch: torch.Tensor(batch_size, seq_len)
    :param sequence_lengths: list(batch_size)
    :return:
    '''
    sequence_lengths = sequence_lengths.cpu().numpy()

    sorted_sequence_lengths = np.sort(sequence_lengths)[::-1]
    sorted_sequence_lengths = torch.from_numpy(sorted_sequence_lengths.copy())

    idx_sort = np.argsort(-sequence_lengths)
    idx_unsort = np.argsort(idx_sort)

    idx_sort = torch.from_numpy(idx_sort)
    idx_unsort = torch.from_numpy(idx_unsort)

    if embedded_sequence_batch.is_cuda:
        idx_sort = idx_sort.cuda()
        idx_unsort = idx_unsort.cuda()

    embedded_sequence_batch = embedded_sequence_batch.index_select(0, idx_sort)

    # # go back to ints as requested by torch (will change in torch 0.4)
    # int_sequence_lengths = [int(elem) for elem in sorted_sequence_lengths.tolist()]
    # Handling padding in Recurrent Networks
    packed_rnn_input = nn.utils.rnn.pack_padded_sequence(embedded_sequence_batch,sorted_sequence_lengths,batch_first=True)
    return packed_rnn_input, idx_unsort

def unpack_rnn_output(packed_rnn_output, indices):
    '''
    :param packed_rnn_output: torch object
    :param indices: Variable(LongTensor) of indices to sort output
    :return:
    '''
    encoded_sequence_batch, _ = nn.utils.rnn.pad_packed_sequence(packed_rnn_output,batch_first=True)
    encoded_sequence_batch = encoded_sequence_batch.index_select(0, indices)

    return encoded_sequence_batch

def mean_pooling(batch_hidden_states, batch_lengths):
    '''
    :param batch_hidden_states: torch.Tensor(batch_size, seq_len, hidden_size)
    :param batch_lengths: list(batch_size)
    :return:
    '''

    batch_lengths = batch_lengths.unsqueeze(1)
    pooled_batch = torch.sum(batch_hidden_states, 1)

    pooled_batch = pooled_batch / batch_lengths.expand_as(pooled_batch).float()

    return pooled_batch


def max_pooling(batch_hidden_states, batch_lengths):
    '''
    :param batch_hidden_states: torch.Tensor(batch_size, seq_len, hidden_size)
    :return:
    '''
    pooled_batch, _ = torch.max(batch_hidden_states, 1)
    return pooled_batch


def gather_last(batch_hidden_states, batch_lengths, bidirectional=True):

    seq_len, batch_size, hidden_x_dirs = batch_hidden_states.size()

    if bidirectional:
        assert hidden_x_dirs % 2 == 0
        single_dir_hidden = int(hidden_x_dirs / 2)
    else:
        single_dir_hidden = int(hidden_x_dirs)

    batch_lengths = batch_lengths.unsqueeze(1).unsqueeze(1)

    fw_batch_lengths = batch_lengths - 1
    fw_batch_lengths = fw_batch_lengths.repeat(1, 1, single_dir_hidden)

    if bidirectional:
        bw_batch_lengths = torch.zeros(*fw_batch_lengths.size()).long()

        if batch_hidden_states.is_cuda:
            bw_batch_lengths = bw_batch_lengths.cuda()

        # we want 2 chunks in the last dimension
        out_fw, out_bw = torch.chunk(batch_hidden_states, 2, 2)

        h_t_fw = torch.gather(out_fw, 1, fw_batch_lengths)
        h_t_bw = torch.gather(out_bw, 1, bw_batch_lengths)

        # -> (batch_size, hidden_x_dirs)
        last_hidden_out = torch.cat([h_t_fw, h_t_bw], 2).squeeze(1)

    else:
        last_hidden_out = \
            torch.gather(batch_hidden_states, 1, fw_batch_lengths).squeeze(1)

    return last_hidden_out
