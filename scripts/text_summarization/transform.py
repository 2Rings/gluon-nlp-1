import sys
import os
import collections
import hashlib
import time
from gluonnlp.data import SpacyTokenizer
from gluonnlp.data import CorpusDataset
from gluonnlp.data import count_tokens
from mxnet.gluon.data import ArrayDataset

def build_vocab(dataset):
    start = time.time()
    data =  CorpusDataset(infile, tokenizer = SpacyTokenizer())
    vocab_counter = count_tokens("")
    vocab_counter = count_tokens(data,counter = vocab_counter)

    return vocab_counter

def trans(dataset, makevocab = False):
    if makevocab:
        vocab = build_vocab(dataset)
    with open(data, 'r') as f:
        lines = f.readlines()

    art = []
    abs = []
    for line in lines:
        line = line.strip('\n').split('\t')
        art.append(line[0])
        abs.append(line[1])

    data = ArrayDataset(art,abs)

    return data, vocab


def process_dataset(dataset, vocab):
    start = time.time()
    pass
