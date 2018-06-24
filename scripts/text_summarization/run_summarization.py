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
from loss import SequenceLoss
import gluonnlp.data.batchify as btf
from encoder_decoder import get_summ_encoder_decoder
from summarization import SummarizationModel
from mxnet import gluon

import transform

parser = argparse.ArgumentParser(description = 'Neural Abstractive Summarization')

parser.add_argument('--dataset', type = str, default = '', help = 'Dataset to use.')
parser.add_argument('--epochs', type = int, default = 2, help = 'upper epoch limit')
parser.add_argument('--mode', type = str, default = 'train', help = 'Train/Validation/Test.')
parser.add_argument('--experiment_name', type = str, default = 'experiment', help = 'experiment name')
parser.add_argument('--hidden_dim', type = int, default = 256, help = 'dimension of RNN hidden states')
parser.add_argument('--embedding_dim', type = int, default = 128, help = 'dimension of word embedding')
parser.add_argument('--batch_size', type = int, default = 5, help = 'Batch Size')
parser.add_argument('--test_batch_size', type = int, default = 16, help = 'Test Batch Size')
parser.add_argument('--max_enc_steps', type = int, default = 5, help = 'max timesteps of encoder (max source text tokens)')
parser.add_argument('--max_dec_steps', type = int, default = 6, help = 'max timesteps of decoder (max summary tokens)')
parser.add_argument('--beam_size', type = int, default = 4, help = 'beam size for beam search decoding.')
parser.add_argument('--min_dec_steps', type = int, default = 35, help = 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
parser.add_argument('--vocab_size', type = int, default = 50000, help = 'Size of vocabulary.')
parser.add_argument('--optimizer', type = str, default = 'adam', help = 'Optimization Algorithm')
parser.add_argument('--lr', type = float, default = 0.15, help = 'Learning rate')
parser.add_argument('--bucket_ratio', type = float, default = 0.0, help = 'bucket_ratio')
parser.add_argument('--num_buckets', type = int, default = 2, help = 'bucket number')
parser.add_argument('--gpu', type = int, default = None, help = 'id of the gpu to use. Set it to empty means to use cpu.')
parser.add_argument('--clip', type = float, default = 3.0, help = 'gradient clipping')

args = parser.parse_args()

# data_path = args.dataset

data_path = "/Users/Admin/Documents/DL/gluon-nlp-1/scripts/text_summarization/finished_files"
train_path = os.path.join(data_path, "train.txt")
val_path = os.path.join(data_path, "val.txt")
test_path = os.path.join(data_path, "test.txt")

data_transform = TrainValDataTransform(max_enc_steps= args.max_enc_steps, max_dec_steps= args.max_dec_steps)

train_data, train_data2idx, my_vocab = data_transform(dataset = train_path, makevocab = True)
val_data, val_data2idx, _ = data_transform(dataset = val_path, vocab = my_vocab)
test_data, test_data2idx, _ = data_transform(dataset = test_path, vocab = my_vocab)

data_train_lengths = [(len(ele[0]), len(ele[1])) for i, ele in enumerate(train_data)]
data_val_lengths = [(len(ele[0]), len(ele[1])) for i, ele in enumerate(val_data)]
data_test_lengths = [(len(ele[0]), len(ele[1])) for i, ele in enumerate(test_data)]


train_data = SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1])) for i, ele in enumerate(train_data)])
val_data = SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i) for i, ele in enumerate(val_data)])
test_data = SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i) for i, ele in enumerate(test_data)])

if args.gpu is None:
    ctx = mx.cpu()
    print('Use CPU')
else:
    ctx = mx.gpu(args.gpu)

encoder, decoder = get_summ_encoder_decoder(hidden_size = args.hidden_dim)

model = SummarizationModel(vocab = my_vocab, encoder = encoder, decoder = decoder, hidden_dim = args.hidden_dim, embed_size = args.embedding_dim, prefix = 'summary_')

loss_function = SequenceLoss(valid_length= args.max_dec_steps, batch_size = args.batch_size, vocab_size = args.vocab_size)
loss_function.initialize(init = mx.init.Uniform(0.1), ctx =ctx)
loss_function.hybridize()
print "#50: Create model"
model = SummarizationModel(vocab = my_vocab, encoder = encoder, decoder = decoder, hidden_dim = args.hidden_dim, embed_size = args.embedding_dim, prefix = 'summary_')

# loss_function = SequenceLoss(valid_length= abs_valid_length, vocab_size = args.vocab_size)
print "#54: Create loss_function"
loss_function = SequenceLoss(valid_length= args.max_dec_steps, batch_size = args.batch_size, vocab_size = args.vocab_size)

print "#55"
loss_function.initialize(init = mx.init.Uniform(0.1), ctx =ctx)
loss_function.hybridize()
print "#56"
model.initialize(init = mx.init.Uniform(0.1), ctx = ctx)
model.hybridize()

# TODO: Summarizer


def evaluate(data_loader):
    summary_out = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0
    for _, (art_seq, abs_seq, art_valid_length, abs_valid_length, inst_ids) in enumerate(data_loader):
        art_seq = art_seq.as_in_context(ctx)
        abs_seq = abs_seq.as_in_context(ctx)
        art_valid_length = art_valid_length.as_in_context(ctx)
        abs_valid_length = abs_valid_length.as_in_context(ctx)

        #Calculating Loss
        out = model(art_seq, abs_seq[:, :-1], art_valid_length, abs_valid_length - 1)
        loss = loss_function(out)
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (abs_seq.shape[1] - 1)
        avg_loss_denom += (abs_seq.shape[1] - 1)

        samples, _, sample_valid_length = summarier.summarize(art_seq = art_seq, art_valid_length = art_valid_length)
        max_score_sample = samples[:, 0, :].asnumpy()
        for i in range(max_score_sample.shape[0]):
            summary_out.append([vocab.idx_to_token[ele] for ele in max_score_sample[i][1:(sample_valid_length[i] - 1)]])

    avg_loss = avg_loss / avg_loss_denom
    real_summary_out = [None for _ in range(all_inst_ids)]
    for ind, sentence in zip(all_inst_ids, summary_out):
        real_translation_out[ind] = sentence

    return avg_loss, real_translation_out

def run_train():
    print "#57: trainer"
    trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate':args.lr})
    print "#58: batchify"

    train_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(), btf.Stack(), btf.Stack())

    test_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(), btf.Stack(), btf.Stack(), btf.Stack())
    print type(train_data[0])
    train_batch_sampler = FixedBucketSampler(lengths = data_train_lengths,
                                                batch_size = args.batch_size,
                                                num_buckets = args.num_buckets,
                                                ratio = args.bucket_ratio,
                                                shuffle = True)
    logging.info('Train Batch Sampler:\n{}'.format(train_batch_sampler.stats()))
    train_data_loader = DataLoader(train_data,
                                    batch_sampler = train_batch_sampler,
                                    batchify_fn=train_batchify_fn,
                                    num_workers=8)

    val_batch_sampler = FixedBucketSampler(lengths = data_val_lengths,
                                                batch_size = args.test_batch_size,
                                                num_buckets = args.num_buckets,
                                                ratio = args.bucket_ratio,
                                                shuffle = False)

    val_data_loader = DataLoader(  val_data,
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

    # art_seq = mx.nd.random.uniform(shape=(5,7))
    # abs_seq = mx.nd.random.uniform(shape=(5,7))
    # art_seq = art_seq.as_in_context(ctx)
    # abs_seq = abs_seq.as_in_context(ctx)
    # art_valid_length = mx.ndarray.ones(shape=(5,))*7
    # abs_valid_length = mx.ndarray.ones(shape=(5,))*7
    # # print art_valid_length
    # art_valid_length = art_valid_length.as_in_context(ctx)
    # abs_valid_length = abs_valid_length.as_in_context(ctx)
    print type(test_data_loader)
    print "trianning loop"
    for epoch_id in range(args.epochs):
        for batch_id, (art_seq, abs_seq, art_valid_length, abs_valid_length) in enumerate(train_data_loader):
            art_seq = art_seq.as_in_context(ctx)
            abs_seq = abs_seq.as_in_context(ctx)
            art_valid_length = art_valid_length.as_in_context(ctx)
            abs_valid_length = abs_valid_length.as_in_context(ctx)
            with mx.autograd.record():
                #out should be our prediction with shape
                decoder_outputs = model(art_seq, abs_seq[:, :-1], art_valid_length, abs_valid_length - 1)
                #decoder_outputs[0] = (batch_size, 2* hidden_dim)
                outs = decoder_outputs[0]
                outs = mx.ndarray.expand_dims(outs, axis = 0)

                for i in range(1, len(decoder_outputs)):
                    ele = mx.ndarray.expand_dims(decoder_outputs[i], axis = 0)
                    outs = mx.ndarray.concat(outs, ele, dim = 0)
                decoder_outputs = outs
                #decoder_outputs : shape(abs_length, bathc_size, 2*hidden_dim)
                abs_seq = abs_seq[:,:-1]
                print abs_seq.shape
                loss = loss_function(decoder_outputs, abs_seq)
                loss.backward()



            trainer.step(args.batch_size)
            grads = [p.grad(ctx) for p in model.collect_params().values()]
            # gnorm = gluon.utils.clip_global_norm(grads, args.clip)


            step_loss = loss.asscalar()

    # valid_loss, valid_translation_out = evaluate(val_data_loader)
    # model.load_params(os.path.join(args.save_dir, 'valid_best.params'))
    # valid_loss, valid_summary = evaluate(val_data_loader)
    # ## TODO: evaluation and rouge

if __name__ == '__main__':
    run_train()
