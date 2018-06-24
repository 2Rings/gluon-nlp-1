import sys
import os
import collections
import hashlib
import time
import gluonnlp
from gluonnlp.data import SpacyTokenizer
from gluonnlp.data import CorpusDataset
from gluonnlp.data import count_tokens
from mxnet.gluon.data import ArrayDataset

embed_type = 'glove'
embd_source = 'glove.6B.50d.txt'

class TrainValDataTransform(object):

    def __init__(self, max_enc_steps, max_dec_steps, embedding_type = 'glove', source = 'glove.6B.50d.txt'):
        self._embedding_type = embedding_type
        self._source = source
        self._max_enc_step = max_enc_steps
        self._max_dec_step = max_dec_steps

    def __call__(self, dataset, makevocab = False, vocab = None):
        if makevocab:
            return self.trans(dataset, makevocab, vocab)
        else:
            return self.trans(dataset, vocab = vocab)

    def get_embedding(self):
        return gluonnlp.embedding.create(self._embedding_type, self._source)

    def build_vocab(self, dataset):
        # Builidding vocabulary
        # my_vocab: embedding, idx_to_token, reserved_token, token_to_idx, unknown_token, padding_token, bos_token, eos_token
        start = time.time()
        data =  CorpusDataset(dataset, tokenizer = SpacyTokenizer())
        vocab_counter = count_tokens("")
        for line in data:
            vocab_counter = count_tokens(line, counter = vocab_counter)

        end = time.time()

        my_vocab = gluonnlp.Vocab(vocab_counter)
        print('Building vocabulary spent: {}'.format(end - start))

        return my_vocab

    def trans(self, dataset, makevocab = False, vocab = None):

        if makevocab:
            my_vocab = self.build_vocab(dataset)
        else:
            my_vocab = vocab

        with open(dataset, 'r') as f:
            lines = f.readlines()

        art = []
        abs = []
        art2idx = []
        abs2idx = []
        for line in lines:
            line = line.strip('\n').split('\t')
            art.append(line[0])
            abs.append(line[1])
            art2idx.append(my_vocab(line[0].split()[:self._max_enc_step]))
            abs2idx.append(my_vocab(line[1].split()[:self._max_dec_step]))


        data = ArrayDataset(art,abs)
        # print data
        data_idx = ArrayDataset(art2idx, abs2idx)
        # print data_idx
        return [data, data_idx, my_vocab]
