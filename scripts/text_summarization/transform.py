import sys
import os
import collections
import hashlib
import time
from gluonnlp.data import SpacyTokenizer
from gluonnlp.data import CorpusDataset
from gluonnlp.data import count_tokens
from mxnet.gluon.data import ArrayDataset

embed_type = 'glove'
embd_source = 'glove.6B.50d.txt'

class TrainValDataTransform(object):

    def __init__(self, dataset, max_enc_steps, max_dec_steps, vocab = None, makevocab = False, embedding_type = 'glove', source = 'glove.6B.50d.txt'):
        self._data = dataset
        self._vocab = vocab
        self._makevocab = makevocab
        self._embedding_type = embedding_type
        self._source = source
        self._max_enc_step = max_enc_steps
        self._max_dec_step = max_dec_steps

    def __call__(self):
        return self.trans()

    def get_embedding(self):
        return gluonnlp.embedding.create(self._embedding_type, self._source)

    def build_vocab(self):
        # Builidding vocabulary
        # my_vocab: embedding, idx_to_token, reserved_token, token_to_idx, unknown_token, padding_token, bos_token, eos_token
        # start = time.time()
        data =  CorpusDataset(self._data, tokenizer = SpacyTokenizer())
        vocab_counter = count_tokens("")
        vocab_counter = count_tokens(data,counter = vocab_counter)
        my_vocab = gluonnlp.Vocab(vocab_counter)
        embd = get_embedding()
        my_vocab = my_vocab.set_embedding(embd)


        return my_vocab

    def trans():
        if makevocab:
            my_vocab = build_vocab()

        with open(data, 'r') as f:
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
        data_idx = ArrayDataset(art2idx, abs2idx)

        return data, data_idx, my_vocab


# def process_dataset(dataset, vocab):
#     start = time.time()
#     pass
