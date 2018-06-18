import numpy as np
import mxnet as mx
# from mxnet.gluon.loss import SoftmaxCELoss


class SequenceLoss(Loss):

    def __init__(self, valid_length, vocab_size, weight = None, batch_axis = 0, **kwargs):
        #axis: the axis to sum over when computing softmax and entropy
        super(SequenceLoss, self).__init__(weight, batch_axis, **kwargs)
        self.valid_length = valid_length
        with self.name_scope():
            self._linear_layer = nn.HybridSequential()
            with self._linear_layer.name_scope():
                self._linear_layer.add(nn.Dense(vocab_size, weight_initializer = mx.init.Uniform(0.1)))
                
    def hybrid_forward(self, decoder_outputs):
        vocab_scores = []
        #decoder_outputs: list of all V[st, ht*] + b

        for i, output in enumerate(decoder_outputs):
            # V'(V[St,ht*]+b)+b'
            #shape(batch_size, Vsize)
            vocab_score = self._linear_layer(output)
            vocab_scores.append(vocab_score)
        #vocab_dists: shape(length, batch_size, vsize)
        vocab_dists = [mx.nd.softmax(v_s) for v_s in vocab_scores]
        batch_num = mx.nd.arange(batch_size, dtype = int)
        loss_per_step = []
        for dec_step, dist in enumerate(vocab_dists):
            targets = mx.nd.array(self.abs_seq[:,dec_step], dtype = int)
            indices = mx.nd.stack((batch_num, targets), axis = 1)
            gold_probs = mx.nd.gather_nd(dist, indices)
            # loss = -mx.nd.log(gold_probs)
            loss = -mx.nd.log(gold_probs)
            loss_per_step.append(loss)

        loss = mx.ndarray.sum(loss_per_step, axis = 1)/self.valid_length

        return mx.ndarray.mean(loss, axis = self._batch_axis, exclude = True)
