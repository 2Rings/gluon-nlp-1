import os
import sys
import io
from time import time
import argparse
import pickle
import random
import logging
import numpy as np
import mxnet as mx
from gluonnlp.data import FixedBucketSampler
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from mxnet.gluon.data import DataLoader
from transform_data import TrainValDataTransform
import gluonnlp.data.batchify as btf
from encoder_decoder import get_summ_encoder_decoder
from summarization import SummarizationModel
from summarization import BeamSearchSummarizer
from gluonnlp.model import BeamSearchSampler, BeamSearchScorer
from mxnet import gluon
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from utils import write_for_rouge, rouge_eval, rouge_log

parser = argparse.ArgumentParser(description = 'Neural Abstractive Summarization')

parser.add_argument('--dataset', type = str, default = '', help = 'Dataset to use.')
parser.add_argument('--epochs', type = int, default = 20, help = 'upper epoch limit')
parser.add_argument('--mode', type = str, default = 'train', help = 'Train/Validation/Test.')
parser.add_argument('--experiment_name', type = str, default = 'experiment_test', help = 'experiment name')
parser.add_argument('--hidden_dim', type = int, default = 128, help = 'dimension of RNN hidden states')
parser.add_argument('--embedding_dim', type = int, default = 64, help = 'dimension of word embedding')
parser.add_argument('--batch_size', type = int, default = 8, help = 'Batch Size')
parser.add_argument('--test_batch_size', type = int, default = 16, help = 'Test Batch Size')
parser.add_argument('--max_enc_steps', type = int, default = 400, help = 'max timesteps of encoder (max source text tokens)')
parser.add_argument('--max_dec_steps', type = int, default = 100, help = 'max timesteps of decoder (max summary tokens)')
parser.add_argument('--beam_size', type = int, default = 4, help = 'beam size for beam search decoding.')
parser.add_argument('--min_dec_steps', type = int, default = 35, help = 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
parser.add_argument('--vocab_size', type = int, default = 100, help = 'Size of vocabulary.')
parser.add_argument('--optimizer', type = str, default = 'AdaGrad', help = 'Optimization Algorithm')
parser.add_argument('--lr', type = float, default = 0.15, help = 'Learning rate')
parser.add_argument('--bucket_ratio', type = float, default = 0.0, help = 'bucket_ratio')
parser.add_argument('--num_buckets', type = int, default = 100, help = 'bucket number')
parser.add_argument('--gpu', type = int, default = None, help = 'id of the gpu to use. Set it to empty means to use cpu.')
parser.add_argument('--clip', type = float, default = 2.0, help = 'gradient clipping')
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--save_dir', type=str, default='out_dir_test', help='directory path to save the final model and training log')
parser.add_argument('--lp_alpha', type=float, default=1.0, help='Alpha used in calculating the length penalty')
parser.add_argument('--lp_k', type=int, default=5, help='K used in calculating the length penalty')
parser.add_argument('--log_root', type=str, default='log', help='log root')
args = parser.parse_args()

# data_path = args.dataset

data_path = "data/finished_files"
train_path = os.path.join(data_path, "train.txt")
val_path = os.path.join(data_path, "val.txt")
test_path = os.path.join(data_path, "test.txt")
decode_dir = os.path.join(args.log_root, args.experiment_name)
rouge_ref_dir = os.path.join(decode_dir, 'reference')
rouge_dec_dir = os.path.join(decode_dir, 'decoded')
data_transform = TrainValDataTransform(max_enc_steps= args.max_enc_steps, max_dec_steps= args.max_dec_steps)

t0 = time()
train_save = data_transform(dataset = train_path, makevocab = True)
val_save = data_transform(dataset = val_path, vocab = train_save[2])
test_save = data_transform(dataset = test_path, vocab = train_save[2])

# ft = open('train_save.pkl', 'wb')
# pickle.dump(train_save, ft, -1)
# ft.close()
#
# fv = open('val_save.pkl', 'wb')
# pickle.dump(val_save, fv, -1)
# fv.close()
#
# ftest = open('test_save.pkl', 'wb')
# pickle.dump(test_save, ftest, -1)
# ftest.close()

logging.info('Processing data spent {}'.format(time() - t0))

train_data_ori, train_data2idx, my_vocab = train_save
val_data_ori, val_data2idx, _ = val_save
test_data_ori, test_data2idx, _ = test_save

data_train_lengths = [(len(ele[0]), len(ele[1])) for i, ele in enumerate(train_data_ori)]
data_val_lengths = [(len(ele[0]), len(ele[1])) for i, ele in enumerate(val_data_ori)]
data_test_lengths = [(len(ele[0]), len(ele[1])) for i, ele in enumerate(test_data_ori)]

train_data = SimpleDataset([(ele[0], ele[1], ele[2], len(ele[0]), len(ele[1]), len(ele[2])) for i, ele in enumerate(train_data2idx)])
val_data = SimpleDataset([(ele[0], ele[1], ele[2], len(ele[0]), len(ele[1]), len(ele[2]), i) for i, ele in enumerate(val_data2idx)])
test_data = SimpleDataset([(ele[0], ele[1], ele[2], len(ele[0]), len(ele[1]), len(ele[2]), i) for i, ele in enumerate(test_data2idx)])

if args.gpu is None:
    ctx = mx.cpu()
    print('Use CPU')
else:
    ctx = mx.gpu(args.gpu)

encoder, decoder = get_summ_encoder_decoder(hidden_size=args.hidden_dim,embedding_size=args.embedding_dim, vocab=my_vocab)

model = SummarizationModel(vocab=my_vocab, encoder=encoder, decoder=decoder, hidden_size=args.hidden_dim, embed_size = args.embedding_dim, prefix = 'summary_')

loss_function = SoftmaxCrossEntropyLoss()
loss_function.initialize(init=mx.init.Uniform(0.02), ctx=ctx)
loss_function.hybridize()
# print "#56"
model.initialize(init=mx.init.Uniform(0.02), ctx=ctx)
model.hybridize()

# TODO: Summarizer
summarizer = BeamSearchSummarizer(model=model,
                                  beam_size=args.beam_size,
                                  scorer=BeamSearchScorer(alpha=args.lp_alpha, K=args.lp_k),
                                  max_length=args.max_dec_steps)



def calc_running_avg_loss(loss, running_avg_loss, decay=0.99):
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss

    return running_avg_loss

def run_eval(data_loader):
    summary_out = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0
    running_avg_loss = 0
    best_loss = None
    for batch_id, (art_seq, abs_seq, abs_target, art_valid_length, abs_valid_length, target_valid_length, inst_ids) in enumerate(data_loader):
        art_seq = art_seq.as_in_context(ctx)
        abs_seq = abs_seq.as_in_context(ctx)
        abs_target = abs_target.as_in_context(ctx)
        art_valid_length = art_valid_length.as_in_context(ctx)
        abs_valid_length = abs_valid_length.as_in_context(ctx)
        target_valid_length = target_valid_length.as_in_context(ctx)
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=art_seq.shape[0], ctx=ctx)
        #Calculating Loss
        ts = time()
        out = model(art_seq, abs_seq, hidden, art_valid_length, abs_valid_length)
        out = mx.ndarray.stack(*out, axis=1)
        loss = loss_function(out, abs_target, target_valid_length).mean().asscalar()
        print("{}: spent {}s and step_loss: {}".format(batch_id, time() - ts, loss))
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (abs_seq.shape[1])
        avg_loss_denom += (abs_seq.shape[1])
        running_avg_loss = calc_running_avg_loss(loss, running_avg_loss)
    avg_loss = avg_loss / avg_loss_denom

    return avg_loss, running_avg_loss

def decode(data_loader):
        summary_out = []
        all_inst_ids = []
        avg_loss_denom = 0
        avg_loss = 0.0
        ex_index = 0
        for batch_id, (art_seq, abs_seq, abs_target, art_valid_length, abs_valid_length, target_valid_length, inst_ids) in enumerate(data_loader):
            art_seq = art_seq.as_in_context(ctx)
            abs_seq = abs_seq.as_in_context(ctx)
            abs_target = abs_target.as_in_context(ctx)
            art_valid_length = art_valid_length.as_in_context(ctx)
            abs_valid_length = abs_valid_length.as_in_context(ctx)
            target_valid_length = target_valid_length.as_in_context(ctx)
            hidden = model.begin_state(func=mx.nd.zeros, batch_size=art_seq.shape[0], ctx=ctx)
            #Calculating Loss
            ts = time()
            out = model(art_seq, abs_seq, hidden, art_valid_length, abs_valid_length)
            out = mx.ndarray.stack(*out, axis=1)
            loss = loss_function(out, abs_target, target_valid_length).mean().asscalar()
            print("{}: spent {}s and step_loss: {}".format(batch_id, time() - ts, loss))
            all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
            avg_loss += loss * (abs_seq.shape[1])
            avg_loss_denom += (abs_seq.shape[1])

            samples, _, sample_valid_length = summarizer.summarize(art_seq = art_seq, art_valid_length=art_valid_length)
            max_score_sample = samples[:, 0, :].asnumpy()
            reference_target = abs_target.asnumpy()
            sample_valid_length = sample_valid_length[:, 0].asnumpy()
            for i in range(max_score_sample.shape[0]):
                best_hyp = [my_vocab.idx_to_token[ele] for ele in max_score_sample[i][1:sample_valid_length[i]]]
                reference_sents = [my_vocab.idx_to_token[ele] for ele in abs_target[i][:sample_valid_length[i]]-1]
                write_for_rouge(reference_sents, best_hyp, rouge_ref_dir, rouge_dec_dir, ex_index)
                ex_index += 1
                summary_out.append(best_hyp)

        avg_loss = avg_loss / avg_loss_denom
        real_summary_out = [None for _ in range(all_inst_ids)]
        for ind, sentence in zip(all_inst_ids, summary_out):
            real_summary_out[ind] = sentence

        return avg_loss, real_summary_out

def run_train():
    # print "#57: trainer"
    trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate':args.lr})
    # print "#58: batchify"
    t0 = time()
    train_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(), btf.Pad(), btf.Stack(), btf.Stack(), btf.Stack())

    test_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(), btf.Pad(), btf.Stack(), btf.Stack(), btf.Stack(), btf.Stack())

    train_batch_sampler = FixedBucketSampler(lengths = data_train_lengths,
                                                batch_size = args.batch_size,
                                                num_buckets = args.num_buckets,
                                                ratio = args.bucket_ratio,
                                                shuffle = False)
    print('Train Batch Sampler:\n{}'.format(train_batch_sampler.stats()))
    train_data_loader = DataLoader(train_data,
                                    batch_sampler = train_batch_sampler,
                                    batchify_fn=train_batchify_fn,
                                    num_workers=8)

    # train_data_loader = DataLoader(train_data,
    #                                batch_size = args.batch_size,
    #                                 #batch_sampler = train_batch_sampler,
    #                                 batchify_fn=train_batchify_fn,
    #                                 num_workers=8)

    val_batch_sampler = FixedBucketSampler(lengths = data_val_lengths,
                                                batch_size = args.test_batch_size,
                                                num_buckets = args.num_buckets,
                                                ratio = args.bucket_ratio,
                                                shuffle = False)

    val_data_loader = DataLoader(val_data,
                                    batch_sampler = val_batch_sampler,
                                    batchify_fn=test_batchify_fn,
                                    num_workers=8)



    print('Batchifying spent {}'.format(time() - t0))

    best_loss = None
    for epoch_id in range(args.epochs):
        log_avg_loss = 0
        running_avg_loss = 0
        log_avg_gnorm = 0
        log_wc = 0
        log_start_time = time()
        save_path = os.path.join(args.save_dir, 'valid_best.params')
        print("epoch_id: {}".format(epoch_id))
        for batch_id, (art_seq, abs_seq, abs_target, art_valid_length, abs_valid_length, target_valid_length) in enumerate(train_data_loader):
            ts = time()
            hidden = model.begin_state(func=mx.nd.zeros, batch_size=art_seq.shape[0], ctx=ctx)
            # print("hidden: ", type(hidden), len(hidden), hidden)
            art_seq = art_seq.as_in_context(ctx)
            abs_seq = abs_seq.as_in_context(ctx)
            abs_target = abs_target.as_in_context(ctx)
            art_valid_length = art_valid_length.as_in_context(ctx)
            abs_valid_length = abs_valid_length.as_in_context(ctx)
            target_valid_length = target_valid_length.as_in_context(ctx)
            # print("target_valid_length: ", target_valid_length.mean())
            with mx.autograd.record():
                #out should be our prediction with shape
                '''
                    SoftmaxCELoss()
                '''
                decoder_outputs = model(art_seq, abs_seq, hidden, art_valid_length, abs_valid_length)
                decoder_outputs = mx.ndarray.stack(*decoder_outputs, axis=1)
                loss = loss_function(decoder_outputs, abs_target)
                loss = loss.mean()

                # print("loss type: ", type(loss), loss)
                # print("target_valid_length: ",  target_valid_length.mean())
                loss = loss * abs_target.shape[1] /target_valid_length.mean()
                loss.backward()

            # grads = [p.grad(ctx) for p in model.collect_params().values()]
            # gnorm = gluon.utils.clip_global_norm(grads, args.clip)
            trainer.step(1)
            step_loss = loss.asscalar()
            #running_avg_loss = calc_running_avg_loss(step_loss, running_avg_loss)
            print ("{}: {}: spent {}s and step_loss: {}".format(epoch_id, batch_id, time() - ts, step_loss))
            art_wc = art_valid_length.sum().asscalar()
            abs_wc = abs_valid_length.sum().asscalar()
            log_avg_loss += step_loss
            # log_avg_gnorm += gnorm
            log_wc += art_wc + abs_wc
            if (batch_id + 1) % args.log_interval == 0:
                wps = log_wc / (time() - log_start_time)
                print('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, gnorm={:.4f}, '
                             'throughput={:.2f}K wps, wc={:.2f}K'
                             .format(epoch_id, batch_id + 1, len(train_data_loader),
                                     log_avg_loss / args.log_interval,
                                     np.exp(log_avg_loss / args.log_interval),
                                     log_avg_gnorm / args.log_interval,
                                     wps / 1000, log_wc / 1000))
                log_start_time = time()
                log_avg_loss = 0
                log_avg_gnorm = 0
                log_wc = 0

            if batch_id%5000 == 0 and batch_id > 0:
                val_avg_loss, running_valid_avg_loss = run_eval(val_data_loader)
                print("val_avg_loss: ", val_avg_loss)
                print("running_valid_avg_loss", running_valid_avg_loss)
                if best_loss is None or running_valid_avg_loss < best_loss:
                    logging.info('Save best parameters to {}'.format(save_path))
                    model.save_params(save_path)
                    best_loss = running_valid_avg_loss

def One_pass_decode():
    test_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(), btf.Pad(), btf.Stack(), btf.Stack(), btf.Stack(), btf.Stack())

    test_batch_sampler = FixedBucketSampler(lengths = data_test_lengths,
                                                batch_size = args.test_batch_size,
                                                num_buckets = args.num_buckets,
                                                ratio = args.bucket_ratio,
                                                shuffle = False)

    test_data_loader = DataLoader(test_data,
                                    batch_sampler = test_batch_sampler,
                                    batchify_fn=test_batchify_fn,
                                    num_workers=8)

    if os.path.isfile(os.path.join(args.save_dir, 'valid_best.params')):
        model.load_params(os.path.join(args.save_dir, 'valid_best.params'))
    else:
        raise Exception("No valid_best.params")

    avg_loss = decode(test_data_loader)

    results_dict = rouge_eval(rouge_ref_dir, rouge_dec_dir)
    rouge_log(results_dict, decode_dir)



if __name__ == '__main__':
    if args.mode == 'train':
        run_train()
    elif args.mode == 'decode':
        One_pass_decode()
