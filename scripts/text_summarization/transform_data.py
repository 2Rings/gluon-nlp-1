import os
import collections
import hashlib
import logging
import time
import gluonnlp
from gluonnlp.data import SpacyTokenizer
from gluonnlp.data import CorpusDataset
from gluonnlp.data import count_tokens
from mxnet.gluon.data import ArrayDataset
import re
import codecs
import pickle
import numpy as np
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

    def build_vocab(self, vocab_file):
        # Builidding vocabulary
        # my_vocab: embedding, idx_to_token, reserved_token, token_to_idx, unknown_token, padding_token, bos_token, eos_token
        print("Building vocabulary...")
        with codecs.open(vocab_file, "rb") as fvocab:
            my_vocab = pickle.loads(fvocab)

        return my_vocab

    def trans(self, dataset, makevocab = False, vocab = None):

        vocabfile = "data/finished_files/my_vocab"
        vocab_exit = os.path.isfile(vocabfile)

        if makevocab:
            if vocab_exit:
                with open(vocabfile, "rb") as fvocab:
                    print(vocabfile)
                    my_vocab = pickle.load(fvocab, encoding="utf-8")
            else:
                raise Exception("please build your vocab first")
        else:
            my_vocab = vocab

        if my_vocab is None:
            raise Exception("my_vocab cannot be None")

        with codecs.open(dataset, 'r', 'utf-8') as f:
            lines = f.readlines()

        start_id = my_vocab.token_to_idx[my_vocab.bos_token]
        stop_id = my_vocab.token_to_idx[my_vocab.eos_token]
        art_list = []
        abs_list = []
        dec_inp_list = []
        dec_target_list = []
        art2idx = []
        # abs2idx = []

        cnt = 0
        t0 = time.time()
        for line in lines:
            #if cnt == 100:
             #   break

            line = line.strip()
            if not re.search('\t', line):
                cnt += 1
                print("{} should have split symbol".format(cnt))
                try:
                    print(line)
                except:
                    print ("ascii")
                continue
            line = line.split('\t')
            art = line[0].split()
            abs = line[1].split()


            art_list.append(art)
            abs_list.append(abs)
            # art2idx.append(my_vocab(art))
            # abs2idx.append(my_vocab(abs))
            art_line = my_vocab[art]
            abs_line = my_vocab[abs]
            if len(art_line) > self._max_enc_step:
                art_line = art_line[:self._max_enc_step]

            if cnt < 2:
                print(art_line)
                print(type(art_line[0]))

            art2idx.append(art_line)

            dec_inp, dec_target = self.get_dec_inp_target_seqs(abs_line, self._max_dec_step,start_id, stop_id)

            dec_inp_list.append(dec_inp)
            dec_target_list.append(dec_target)

            if cnt%5000 == 0:
                print("{} have processed".format(cnt))

            cnt += 1

        print("Transform data spent {}".format(time.time() - t0))

        data = ArrayDataset(art_list,abs_list)
        # print data
        art2idx = np.array(art2idx)
        dec_inp_list = np.array(dec_inp_list)
        dec_target_list = np.array(dec_target_list)
        print("data_transform", art2idx.shape, dec_inp_list.shape, dec_target_list.shape)
        data_idx = ArrayDataset(art2idx, dec_inp_list, dec_target_list)
        # print data_idx
        print("my_vocab_sive:", len(my_vocab))
        return data, data_idx, my_vocab

    def get_dec_inp_target_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:
            inp = inp[:max_len]
            target = target[:max_len]
        else:
            target.append(stop_id)
        assert len(inp) == len(target)
        return inp, target
