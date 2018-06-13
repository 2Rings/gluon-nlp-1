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

def get_embedding(embedding_type = 'glove', source = 'glove.6B.50d.txt'):
    return gluonnlp.embedding.create(embedding_type, source)

def build_vocab(dataset):
    # Builidding vocabulary
    # my_vocab: embedding, idx_to_token, reserved_token, token_to_idx, unknown_token, padding_token, bos_token, eos_token
    start = time.time()
    data =  CorpusDataset(infile, tokenizer = SpacyTokenizer())
    vocab_counter = count_tokens("")
    vocab_counter = count_tokens(data,counter = vocab_counter)
    my_vocab = gluonnlp.Vocab(vocab_counter)
    embd = get_embedding(embed_type, embd_source)
    my_vocab = my_vocab.set_embedding(embd)


    return my_vocab

def trans(dataset, makevocab = False):
    if makevocab:
        my_vocab = build_vocab(dataset)
    with open(data, 'r') as f:
        lines = f.readlines()

    art = []
    abs = []
    for line in lines:
        line = line.strip('\n').split('\t')
        art.append(line[0])
        abs.append(line[1])

    data = ArrayDataset(art,abs)

    return data, my_vocab


# def process_dataset(dataset, vocab):
#     start = time.time()
#     pass
