import os
import json
import time
import math
import torch
import pickle
import pdb
import numpy as np
from random import shuffle
from torch import tensor

from torch.functional import Tensor

from ExpSDN.utils.vocab import Vocab
from ExpSDN.utils.sentence import get_embedding_matrix
from ExpSDN.utils import rnns

from torch.utils.data import Dataset


from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

import extension as ext
from extension.utils_tlg import io_utils
from torch.utils.data.distributed import DistributedSampler


def create_loaders(loader_configs):
    dsets, L = {}, {}
    for di,dt in enumerate(loader_configs.keys()):
        shuffle = True if dt == "train" else False
        drop_last = True if dt == "train" else False
        dsets[dt] = CHARADES_STA(loader_configs[dt])
        if ext.distributed.get_world_size() > 1:
            loader_sampler = DistributedSampler(dsets[dt],  num_replicas=ext.distributed.get_world_size(), rank=ext.distributed.get_rank(), shuffle=shuffle)
            shuffle = False
        else:
            loader_sampler = None
        L[dt] = torch.utils.data.DataLoader(
            dsets[dt],
            batch_size = loader_configs[dt]["batch_size"],
            num_workers = loader_configs[dt]["num_workers"],
            shuffle = shuffle, # shuffle
            collate_fn = dsets[dt].collate_fn,
            sampler = loader_sampler,
            drop_last= drop_last #drop_last
        )
    return dsets, L

class CHARADES_STA(Dataset):

    def __init__(self, config):

        self.feature_path = config['features_path']
        self.ann_file_path = config['ann_file_path']
        self.embeddings_path = config['embeddings_path']
        self.data_dir = config['data_dir']
        self.min_count = config['min_count']
        self.train_max_length = config['train_max_length']
        self.test_max_length = config['test_max_length']
        self.feature_sample_num = config["feature_sample_num"]

        self.embeddings_file_path = os.path.join(self.data_dir, f'charades_embeddings_{self.min_count}_{self.train_max_length}.pth') 
        self.vocab_file_path = os.path.join(self.data_dir, f'charades_vocab_{self.min_count}_{self.train_max_length}.pickle')
        

        self.is_training = 'train' in config['split']
        self.i3dfeat = None

        print(self.is_training)

        print('loading annotations into memory...', end=" ")
        tic = time.time()
        
        aux = json.load(open(self.ann_file_path, 'r'))

        self.dataset = aux['annotations']

        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        
        self.create_vocab()
        
        self.get_embedding_matrix(self.embeddings_path)

        self.createIndex()
        self.ids   = list(self.anns.keys())
        self.epsilon = 1E-10


    def create_vocab(self):
        print(self.vocab_file_path, os.path.exists(self.vocab_file_path))
        if self.is_training:
            if not os.path.exists(self.vocab_file_path):
                print("Creating vocab")
                self.vocab = Vocab(
                    add_bos=False,
                    add_eos=False,
                    add_padding=False,
                    min_count=self.min_count)

                for example in self.dataset:
                    self.vocab.add_tokenized_sentence(example['tokens'][:self.train_max_length])

                self.vocab.finish()

                # with open(self.vocab_file_path, 'wb') as f:
                #     pickle.dump(self.vocab, f)
                io_utils.write_pkl(self.vocab_file_path, self.vocab)
            else:
                # with open(self.vocab_file_path, 'rb') as f:
                #     self.vocab = pickle.load(f)
                self.vocab = io_utils.load_pkl(self.vocab_file_path)

        else:
            print("Cargando vocab")
            with open(self.vocab_file_path, 'rb') as f:
                self.vocab = pickle.load(f)


    def get_embedding_matrix(self, embeddings_path):
        '''
        Gets you a torch tensor with the embeddings
        in the indices given by self.vocab.

        Unknown (unseen) words are each mapped to a random,
        different vector.


        :param embeddings_path:
        :return:
        '''
        if self.is_training and not os.path.exists(self.embeddings_file_path):
            tic = time.time()

            print('loading embeddings into memory...', end=" ")

            if 'glove' in embeddings_path.lower():
                tmp_file = get_tmpfile("test_word2vec.txt")
                _ = glove2word2vec(embeddings_path, tmp_file)
                embeddings = KeyedVectors.load_word2vec_format(tmp_file)
            else:
                embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)

            print('Done (t={:0.2f}s)'.format(time.time() - tic))

            embedding_matrix = get_embedding_matrix(embeddings, self.vocab)

            with open(self.embeddings_file_path, 'wb') as f:
                torch.save(embedding_matrix, f)

        else:
            tic = time.time()
            print(f'loading embedding_matrix from {self.embeddings_file_path}...', end=" ")
            with open(self.embeddings_file_path, 'rb') as f:
                embedding_matrix  = torch.load(f)
            print('Done (t={:0.2f}s)'.format(time.time() - tic))

        self.embedding_matrix = embedding_matrix


    def createIndex(self):
        print("Creating index..", end=" ")
        anns = {}
        size = int(round(len(self.dataset) * 1.))
        counter = 0
        for row in self.dataset[:size]:
            if float(row['feature_start']) > float(row['feature_end']):
                print(row)
                continue

            if math.floor(float(row['feature_end'])) >= float(row['number_features']):
                row['feature_end'] = float(row['number_features'])-1

            row['augmentation'] = 0
            anns[counter] = row
            counter+=1
    
        self.anns = anns
        print(" Ok! {}".format(len(anns.keys())))

    def __getitem__(self, index):
        ann = self.anns[index]

        
        if self.i3dfeat is None:
            self.i3dfeat = io_utils.load_hdf5(self.feature_path, verbose=False)
        ann = self.anns[index]
        # print(ann)

        i3dfeat = self.i3dfeat[ann['video']][:]
        i3dfeat = torch.from_numpy(i3dfeat).float()

        # origin_feat_length = i3dfeat.shape[0]
        feat_length = i3dfeat.shape[0]

        

        if self.is_training:
            raw_tokens = ann['tokens'][:self.train_max_length]
        else:
            raw_tokens = ann['tokens'][:self.test_max_length]
        # print(ann['tokens'])
        indices = self.vocab.tokens2indices(raw_tokens) #change words to index
        tokens = [self.embedding_matrix[index] for index in indices] #get word embedding by looking up the GloVe table
        tokens = torch.stack(tokens) #get origin query vector

        slice_num = 1024 if self.feature_sample_num < 0 else self.feature_sample_num
        if i3dfeat.shape[0] > slice_num:
            idx = np.linspace(0, i3dfeat.shape[0]-1, num=slice_num, dtype = int)
            i3dfeat = i3dfeat[idx]
            ann['feature_start'] = ann['feature_start'] * (slice_num/ann['number_features'])
            ann['feature_end'] = ann['feature_end'] * (slice_num/ann['number_features'])
            ann['number_features'] = slice_num
            if ann['feature_start'] == slice_num:
                ann['feature_start'] -= 1
            if ann['feature_end'] == slice_num:
                ann['feature_end'] -= 1
        feat_length = i3dfeat.shape[0]


        localization = np.zeros(feat_length, dtype=np.float32)
        start = math.floor(ann['feature_start'])
        end   = math.floor(ann['feature_end'])
        time_start = ann['time_start']
        time_end = ann['time_end']

        factor = ann['number_frames']/ann['number_features']


        loc_start = np.ones(feat_length, dtype=np.float32) * self.epsilon
        loc_end   = np.ones(feat_length, dtype=np.float32) * self.epsilon

        y = 0
        y_2 = (1 - (feat_length-5) * self.epsilon - y)/ 9
        y_1 = y_2 * 2
        y_0 = y_2 * 3

        if start > 0:
            loc_start[start - 1] = y_1
        if start > 1:
            loc_start[start - 2] = y_2
        if start < feat_length-1:
            loc_start[start + 1] = y_1
        if start < feat_length-2:
            loc_start[start + 2] = y_2
        loc_start[start] = y_0
        
        if end > 0:
            loc_end[end - 1] = y_1
        if end > 1:
            loc_end[end - 2] = y_2
        if end < feat_length-1:
            loc_end[end + 1] = y_1
        if end < feat_length-2:
            loc_end[end + 2] = y_2
        loc_end[end] = y_0

        loc_offset = np.arange(0, feat_length) * factor
        start_loc_offset = ((time_start * ann['fps'] - loc_offset)/ann['number_frames']).astype(np.float32)
        end_loc_offset =((time_end * ann['fps'] - loc_offset)/ann['number_frames']).astype(np.float32)

        y = 1.0 
        localization[start:end] = y


        return index, i3dfeat, None, None, tokens, torch.from_numpy(loc_start), torch.from_numpy(loc_end), \
               torch.from_numpy(localization), time_start, time_end, factor, ann['fps'], raw_tokens, torch.from_numpy(start_loc_offset), torch.from_numpy(end_loc_offset), ann['number_frames']

    def collate_fn(self, batch):
        transposed_batch = list(zip(*batch))

        index      = transposed_batch[0]
        videoFeat  = transposed_batch[1]
        objectFeat = transposed_batch[2]
        humanFeat  = transposed_batch[3]
        tokens     = transposed_batch[4]
        start      = transposed_batch[5]
        end        = transposed_batch[6]
        localiz    = transposed_batch[7]
        time_start = transposed_batch[8]
        time_end   = transposed_batch[9]
        factor     = transposed_batch[10]
        fps        = transposed_batch[11]
        raw_tokens = transposed_batch[12]

        start_offset = transposed_batch[13]
        end_offset = transposed_batch[14]
        num_frame = transposed_batch[15]


        videoFeat, videoFeat_lengths = rnns.pad_sequence(videoFeat, instant_padding=False, padding_num=256)
        

        localiz, localiz_lengths = rnns.pad_sequence(localiz, instant_padding=False, padding_num=256)

        tokens, tokens_lengths   = rnns.pad_sequence(tokens)

        start, start_lengths = rnns.pad_sequence(start)
        end, end_lengths     = rnns.pad_sequence(end)

        return {'index':index, #pair's index
                'raw_tokens': raw_tokens,
               'tokens': tokens, #padded sentence vector, Tensor [B, L], L depends on the longest query vector
               'tokens_lengths': tokens_lengths,  #origin length
            #    'start': torch.tensor(start),  #initial distribution, Tensor[B, T], T depends on the longest video
            #    'end': torch.tensor(end), #distribution, the same as above
                'start': start,  #initial distribution, Tensor[B, T], T depends on the longest video,
               'end': end, #distribution, the same as above
               'localiz': localiz, #padded frame loc mask(0 or 1), used for attention
               'localiz_lengths': localiz_lengths,  #origin length 
               'time_start':  torch.tensor(time_start), # time
               'time_end':  torch.tensor(time_end), 
               'factor': torch.tensor(factor), # video length/T, ann['number_frames']/ann['number_features']
               'fps': torch.tensor(fps), 

               'videoFeat': videoFeat, 
               'videoFeat_lengths': videoFeat_lengths, 

        }

    def __len__(self):
        return len(self.ids)
