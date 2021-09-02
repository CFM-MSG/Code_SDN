import os
import json
import time
import math
import torch
import pickle
import pdb
import numpy as np


from torch.functional import Tensor
import torchtext

from ExpSDN.utils.vocab import Vocab
from ExpSDN.utils.sentence import get_embedding_matrix
from ExpSDN.utils import rnns

from torch.utils.data import Dataset

# import torchtext

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
        dsets[dt] = TACOS(loader_configs[dt])
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

class TACOS(Dataset):

    def __init__(self, config):

        self.feature_path = config['features_path']
        self.ann_file_path = config['ann_file_path']
        self.embeddings_path = config['embeddings_path']
        self.data_dir = config['data_dir']
        self.min_count = config['min_count']
        self.train_max_length = config['train_max_length']
        self.test_max_length = config['test_max_length']
        self.feature_sample_num = config["feature_sample_num"]

        self.embeddings_file_path = os.path.join(self.data_dir, f'tacos_embeddings_{self.min_count}_{self.train_max_length}.pth') 
        self.vocab_file_path = os.path.join(self.data_dir, f'tacos_vocab_{self.min_count}_{self.train_max_length}.pickle')
        
        self.vocab = torchtext.vocab.pretrained_aliases['glove.840B.300d'](cache=config['embeddings_path'])
        self.vocab.itos.extend(['<unk>'])
        self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
        self.vocab.vectors = torch.cat([self.vocab.vectors, torch.zeros(1, self.vocab.dim)], dim=0)
        self.word_embedding = torch.nn.Embedding.from_pretrained(self.vocab.vectors)

        self.is_training = 'train' in config['split']
        self.i3dfeat = None
        print(self.is_training)

        print('loading annotations into memory...', end=" ")
        tic = time.time()
        
        aux = json.load(open(self.ann_file_path, 'r'))

        self.dataset = aux['annotations']
        
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        
        # self.create_vocab()
        
        # self.get_embedding_matrix(self.embeddings_path)

        self.createIndex()
        # pdb.set_trace()
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
                print("generating tmp_file...")
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

            # if self.is_training:


            #     if float(row['number_features']) < 10:
            #         continue            # print(row) 
            #     if float(row['number_features']) >= 1200:
            #         continue            # print(row)
            if float(row['feature_start']) > float(row['feature_end']):
                # print(row)
                continue

            if math.floor(float(row['feature_end'])) >= float(row['number_features']):
                row['feature_end'] = float(row['number_features'])-1

            if self.is_training:

                row['augmentation'] = 1
                anns[counter] = row.copy()
                counter += 1
                continue

            row['augmentation'] = 0
            anns[counter] = row
            counter+=1
        self.anns = anns

        print(" Ok! {}".format(len(anns.keys())))

    def __getitem__(self, index):
        if self.i3dfeat is None:
            self.i3dfeat = io_utils.load_hdf5(self.feature_path, verbose=False)
        ann = self.anns[index]
        # print(ann)

        i3dfeat = self.i3dfeat[ann['video']][:]
        i3dfeat = torch.from_numpy(i3dfeat)

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

        factor = ann['number_frames']/ann['number_features']


        feat_length = i3dfeat.shape[0]

        if self.is_training:
            raw_tokens = ann['tokens'][:self.train_max_length]
        else:
            raw_tokens = ann['tokens'][:self.test_max_length]

        # indices = self.vocab.tokens2indices(raw_tokens)
        # tokens = [self.embedding_matrix[index] for index in indices]
        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 1438) for w in raw_tokens], dtype=torch.long)
        tokens = self.word_embedding(word_idxs)

        # tokens = torch.stack(tokens)

        if ann['augmentation'] == 2: #disable the useless augmentation
            feature_start = ann['feature_start']
            feature_end   = ann['feature_end']

            offset = int(math.floor(feature_start))
            if offset != 0:
                offset = np.random.randint(0, int(round(feature_start)))

            new_feature_start = feature_start - offset
            new_feature_end   = feature_end - offset

            i3dfeat = i3dfeat[offset:,:]

            feat_length = ann['number_features'] - offset
            localization = np.zeros(feat_length, dtype=np.float32)

            start = math.floor(new_feature_start)
            end   = math.floor(new_feature_end)

            time_start = (new_feature_start * ann['number_frames']/ann['number_features']) / ann['fps']
            time_end = (new_feature_end * ann['number_frames']/ann['number_features']) / ann['fps']
            time_offset = (offset * ann['number_frames']/ann['number_features']) / ann['fps']


        else:
            localization = np.zeros(feat_length, dtype=np.float32)

            # loc_start =
            start = math.floor(ann['feature_start'])
            end   = math.floor(ann['feature_end'])
            time_start = ann['time_start']
            time_end = ann['time_end']


        loc_start = np.ones(feat_length, dtype=np.float32) * self.epsilon
        loc_end   = np.ones(feat_length, dtype=np.float32) * self.epsilon

        y = 0
        y_2 = (1 - (feat_length-5) * self.epsilon - y)/ 9
        y_1 = y_2 * 2
        y_0 = y_2 * 3
        # y_1 = (1 - (feat_length-3) * self.epsilon - y)/ 4
        # y_0 = y_1*2


        if start > 0:
            loc_start[start - 1] = y_1
        if start > 1:
            loc_start[start - 2] = y_2
        if start < feat_length-1:
            loc_start[start + 1] = y_1
        if start < feat_length-2:
            loc_start[start + 2] = y_2
        loc_start[start] = y_0
        
        # loc_start = np.zeros(feat_length, dtype=np.float32) * self.epsilon
        # loc_end   = np.zeros(feat_length, dtype=np.float32) * self.epsilon

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
        # start_loc_offset = np.exp(start_loc_offset, dtype=np.float32)
        # end_loc_offset = np.exp(end_loc_offset, dtype=np.float32)


        y = 1.0 
        # loc_start[start] = y
        # loc_end[end] = y
        localization[start:end+1] = y
        # print(time_start, time_end)

        # loc_start = np.exp(start/feat_length, dtype=np.float32)
        # loc_end = np.exp(end/feat_length, dtype=np.float32)        


        # return index, i3dfeat, object_features, human_features, tokens, loc_start, loc_end, \
        #        torch.from_numpy(localization), time_start, time_end, ann['number_frames']/ann['number_features'], ann['fps'], raw_tokens

        return index, i3dfeat, None, None, tokens, torch.from_numpy(loc_start), torch.from_numpy(loc_end), \
               torch.from_numpy(localization), time_start, time_end, factor, ann['fps'], raw_tokens, torch.from_numpy(start_loc_offset), torch.from_numpy(end_loc_offset), ann['number_frames']

        # return index, i3dfeat, None, None, tokens, loc_start, loc_end, \
        #        torch.from_numpy(localization), time_start, time_end, ann['number_frames']/ann['number_features'], ann['fps'], raw_tokens


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

        # regionFeat = humanFeat + objectFeat
        # regionFeat, regionFeat_lengths = rnns.pad_spatial_sequence(regionFeat)

        # objectFeat, objectFeat_lengths = rnns.pad_spatial_sequence(objectFeat)

        # humanFeat, humanFeat_lengths = rnns.pad_spatial_sequence(humanFeat)      



        videoFeat, videoFeat_lengths = rnns.pad_sequence(videoFeat, instant_padding=False, padding_num=256)
        

        localiz, localiz_lengths = rnns.pad_sequence(localiz, instant_padding=False, padding_num=256)

        tokens, tokens_lengths   = rnns.pad_sequence(tokens)

        start, start_lengths = rnns.pad_sequence(start)
        end, end_lengths     = rnns.pad_sequence(end)

        # start_offset, _ = rnns.pad_sequence(start_offset)
        # end_offset, _ = rnns.pad_sequence(end_offset)

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

            #    'start_offset': start_offset,
            #    'end_offset': end_offset,
            #    'num_frame': torch.tensor(num_frame),

               'videoFeat': videoFeat, 
               'videoFeat_lengths': videoFeat_lengths, 
            #    'objectFeat': objectFeat, 
            #    'objectFeat_lengths': objectFeat_lengths, 
            #    'humanFeat': humanFeat, 
            #    'humanFeat_lengths': humanFeat_lengths,
            #    'regionFeat': regionFeat,
            #    'regionFeat_lengths': regionFeat_lengths
        }

    def __len__(self):
        return len(self.ids)
