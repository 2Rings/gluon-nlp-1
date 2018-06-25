import numpy as np
import mxnet as mx
from mxnet.gluon.loss import Loss
from mxnet.gluon import nn, rnn


class SequenceLoss(Loss):

    def __init__(self, valid_length, vocab_size, batch_size = 5, weight = None, batch_axis = 0, **kwargs):
        #axis: the axis to sum over when computing softmax and entropy
        super(SequenceLoss, self).__init__(weight, batch_axis, **kwargs)
        self.valid_length = valid_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.batch_axis = batch_axis
        with self.name_scope():
            self._linear_layer = nn.HybridSequential()
            with self._linear_layer.name_scope():
                self._linear_layer.add(nn.Dense(self.vocab_size, weight_initializer = mx.init.Uniform(0.1)))

    def hybrid_forward(self, F, decoder_outputs, abs_seq):
        vocab_scores = []
        print type(decoder_outputs), type(abs_seq)
        #decoder_outputs: list of all V[st, ht*] + b
        for i, output in enumerate(decoder_outputs):
            # V'(output)+b' => shape(batch_size, Vsize)
            vocab_score = self._linear_layer[0](output)
            vocab_scores.append(vocab_score)
        #vocab_dists: shape(length, batch_size, vsize)
        vocab_dists = [F.softmax(v_s) for v_s in vocab_scores]
        batch_num = F.arange(self.batch_size, dtype = int)
        loss_per_step = []
        abs_seq = F.split(abs_seq, num_outputs = 6,axis = 1)
        for dec_step, dist in enumerate(vocab_dists):
            targets = F.cast(abs_seq[dec_step], dtype = int)
            indices = F.stack(batch_num, targets, axis = 1)
            gold_probs = F.gather_nd(dist, indices)
            loss = -F.log(gold_probs)
            loss_per_step.append(loss)

        loss_per_step = mx.symbol.Group(loss_per_step)
        loss = F.sum(loss_per_step, axis = 0)
        return F.mean(loss, axis = self.batch_axis, exclude = True)
