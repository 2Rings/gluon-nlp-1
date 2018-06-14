import sys
import os
import io
import collections
import hashlib
import time
import argparse
import random
import logging
import numpy as np
import mxnet as mx
from gluonnlp.data import FixedBucketSampler
from gluonnlp.data import SpacyTokenizer
from gluonnlp.data import CorpusDataset
from gluonnlp.data import count_tokens
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from mxnet.gluon.data import DataLoader
from transform import TrainValDataTransform

import transform

parser = argparse.AugmentParser(description = 'Neural Abstractive Summarization')

parser.add_argument('--dataset', type = str, default = '', help = 'Dataset to use.')
parser.add_argument('--epochs', type = int, default = 40, help = 'upper epoch limit')
parser.add_argument('--mode', type = str, default = 'train', help = 'Train/Validation/Test.')
parser.add_argument('--experiment_name', type = str, default = 'experiment', help = 'experiment name')
parser.add_argument('--hidden_dim', type = int, default = 256, help = 'dimension of RNN hidden states')
parser.add_argument('--embedding_dim', type = int, default = 128, help = 'dimension of word embedding')
parser.add_argument('--batch_size', type = int, default = 16, help = 'Batch Size')
parser.add_argument('--test_batch_size', type = int, default = 16, help = 'Test Batch Size')
parser.add_argument('--max_enc_steps', type = int, default = 400, help = 'max timesteps of encoder (max source text tokens)')
parser.add_argument('--max_dec_steps', type = int, default = 100, help = 'max timesteps of decoder (max summary tokens)')
parser.add_argument('--beam_size', type = int, default = 4, help = 'beam size for beam search decoding.')
parser.add_argument('--min_dec_steps', type = int, default = 35, help = 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
parser.add_argument('--vocab_size', type = int, default = 50000, help = 'Size of vocabulary.')
parser.add_argument('--optimizer', type = str, default = 'adam', help = 'Optimization Algorithm')
parser.add_argument('--lr', type = float, default = 0.15, help = 'Learning rate')
parser.add_argument('--bucket_ratio', type = float, default = 0.0, help = 'bucket_ratio')
parser.add_argument('--num_buckets', type = int, default = 8, help = 'bucket number')
args = parser.parse_args()
print(args)
data_path = args.dataset

train_path = os.path.join(data_path, "train.story")
val_path = os.path.join(data_path, "val.story")
test_path = os.path.join(data_path, "test.stroy")


# train_data, train_data2idx, my_vocab = transform.trans(train_data, makevocab = True)
# val_data, val_data2idx, _ = transform.trans(val_path, vocab = my_vocab)
# test_data, test_data2idx, _ = transform.trans(test_path, vocab = my_vocab)

train_data, train_data2idx, my_vocab = TrainValDataTransform(train_data, args.max_enc_steps, args.max_dec_steps, makevocab = True)
val_data, val_data2idx, _ = TrainValDataTransform(val_path, args.max_enc_steps, args.max_dec_steps, vocab = my_vocab)
test_data, test_data2idx, _ = TrainValDataTransform(test_path, args.max_enc_steps, args.max_dec_steps, vocab = my_vocab)

train_data = SimpleDataset([(ele[0], ele[1], len(ele[0]), len(len[1])) for i, ele in enumerate(train_data)])
val_data = SimpleDataset([(ele[0], ele[1], len(ele[0]), len(len[1]), i) for i, ele in enumerate(val_data)])
test_data = SimpleDataset([(ele[0], ele[1], len(ele[0]), len(len[1]), i) for i, ele in enumerate(test_data)])


# loss_function = SoftmaxCEMaskedLoss()
# loss_function.hybridize()

def run_train():
    trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate':args.lr})

    train_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(), btf.Stack(), btf.Stack())

    test_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(), btf.Stack(), btf.Stack(), btf.Stack())

    train_batch_sampler = FixedBucketSampler(lengths = data_train_lengths,
                                                batch_size = args.batch_size,
                                                num_buckets = args.num_buckets,
                                                ratio = args.bucket_ratio,
                                                shuffle = True)

    train_data_loader = DataLoader(train_data,
                                    batch_sampler = train_batch_sampler,
                                    batchify_fn=train_batchify_fn,
                                    num_workers=8)

    val_batch_sampler = FixedBucketSampler(lengths = data_val_lengths,
                                                batch_size = args.test_batch_size,
                                                num_buckets = args.num_buckets,
                                                ratio = args.bucket_ratio,
                                                shuffle = False)

    val_data_loader = DataLoader(val_data,
                                    batch_sampler = val_batch_sampler,
                                    batchify_fn=test_batchify_fn,
                                    num_workers=8)

    test_batch_sampler = FixedBucketSampler(lengths = data_test_lengths,
                                                batch_size = args.test_batch_size,
                                                num_buckets = args.num_buckets,
                                                ratio = args.bucket_ratio,
                                                shuffle = False)

    test_data_loader = DataLoader(test_data,
                                    batch_sampler = test_batch_sampler,
                                    batchify_fn=test_batchify_fn,
                                    num_workers=8)

    for epoch_id in range(args.epochs):
        for batch_id, (art_seq, abs_seq, art_valid_length, abs_valid_length) in enumerate(train_data_loader):
            art_seq = art_seq.as_in_context(ctx)
            abs_seq = abs_seq.as_in_context(ctx)
            art_valid_length = art_valid_length.as_in_context(ctx)
            abs_valid_length = abs_valid_length.as_in_context(ctx)
            with mx.autograd.record():
                #out should be our prediction with shape
                decoder_states = model(art_seq, abs_seq[:, :-1], art_valid_length, abs_valid_length - 1)
                vocab_scores = []
                for i, output in enumerate(decoder_outputs):
                    # V'(V[St,ht*]+b)+b'
                    #shape(batch_size, Vsize)
                    vocab_score = linear(output, weight, b)
                    vocab_scores.append(vocab_score)
                #vocab_dists: shape(length, batch_size, vsize)
                vocab_dists = [mx.nd.softmax(v_s) for v_s in vocab_scores]
                batch_num = mx.nd.arange(batch_size, dtype = int)
                loss_per_step = []
                for dec_step, dist in enumerate(vocab_dists):
                    targets = mx.nd.array(self.abs_seq[:,dec_step], dtype = int)
                    indices = mx.nd.stack((batch_num, targets), axis = 1)
                    gold_probs = mx.nd.gather_nd(dist, indices)
                    loss = -mx.nd.log(gold_probs)
                    loss_per_step.append(loss)

                loss = sum(loss_per_step)/abs_valid_length
                loss = self._loss.mean()
                loss.backward()


            grads = [p.grad(ctx) for p in model.collect_params().values]
            gnorm = gluon.utils.clip_global_norm(grads, args.clip)

        # valid_loss, valid_translation_out = evaluate(val_data_loader)
    model.load_params(os.path.join(args.save_dir, 'valid_best.params'))
    valid_loss, valid_summary = evaluate(val_data_loader)
    ## TODO: evaluation and rouge




    pass

if __name__ == '__main__':
    run_train()
