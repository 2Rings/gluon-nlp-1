import numpy as np
import mxnet as mx
from mxnet.gluon.loss import Loss
from mxnet.gluon import nn, rnn
# from mxnet.gluon.loss import SoftmaxCELoss


class SequenceLoss(Loss):

    def __init__(self, valid_length, vocab_size, batch_size = 5, weight = None, batch_axis = 0, **kwargs):
        #axis: the axis to sum over when computing softmax and entropy
        super(SequenceLoss, self).__init__(weight, batch_axis, **kwargs)
        self.valid_length = valid_length
        self.batch_size = batch_size
<<<<<<< HEAD
        self.vocab_size = vocab_size
        with self.name_scope():
            self._linear_layer = nn.HybridSequential()
            with self._linear_layer.name_scope():
                self._linear_layer.add(nn.Dense(self.vocab_size, weight_initializer = mx.init.Uniform(0.1)))
=======
        print "#55: loss linear layer"
        #linear layer for V'Input + b, project inputs into Vsize
        with self.name_scope():
            self._linear_layer = nn.HybridSequential()
            with self._linear_layer.name_scope():
                self._linear_layer.add(nn.Dense(vocab_size, weight_initializer = mx.init.Uniform(0.1)))

    # def forward(self, decoder_outputs, abs_seq):
    #     return self.hybrid_forward(F, decoder_outputs, abs_seq)
>>>>>>> c727e446161eb1da907678319584c94d8dee0c58

    def hybrid_forward(self, F, decoder_outputs, abs_seq):
        vocab_scores = []
        print "F: ", F
        #decoder_outputs: list of all V[st, ht*] + b
        for i, output in enumerate(decoder_outputs):
            # V'(input)+b'
            #shape(batch_size, Vsize)
<<<<<<< HEAD
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
=======
            print "output: ", type(output)
            vocab_score = self._linear_layer[0](output)
            vocab_scores.append(vocab_score)
        #vocab_dists: shape(length, batch_size, vsize)
        print "vocab_dists: ", type(vocab_scores), type(vocab_scores[0])
        print vocab_score
        vocab_dists = [F.softmax(v_s) for v_s in vocab_scores]
        batch_num = F.arange(self.batch_size, dtype = int)
        loss_per_step = []
        abs_seq = F.split(abs_seq, num_outputs = 7,axis = 1)
        for dec_step, dist in enumerate(vocab_dists):
            print "dec_step: ", type(dec_step)
            # target = F.pick(abs_seq, index = [ dec_step ], 0)
            targets = F.cast(abs_seq[dec_step], dtype = int)
            print "#100"
            indices = F.stack(batch_num, targets, axis = 1)
            print "#101"
            gold_probs = F.gather_nd(dist, indices)
            print "#102"
            # loss = -mx.nd.log(gold_probs)
            loss = -F.log(gold_probs)
            print "#103"
            loss_per_step.append(loss)

        print "#104"
        loss_per_step = mx.symbol.Group(loss_per_step)
        loss = F.sum(loss_per_step, axis = 0)
        print "#105"
>>>>>>> c727e446161eb1da907678319584c94d8dee0c58
        return F.mean(loss, axis = self._batch_axis, exclude = True)
