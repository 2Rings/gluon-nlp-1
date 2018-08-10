import numpy as np
from mxnet.gluon import Block
from mxnet.gluon import nn, rnn
import mxnet as mx
from gluonnlp.model import BeamSearchSampler, BeamSearchScorer
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
__all__ = ['SummarizationModel']


class SummarizationModel(Block):

    def __init__(self, vocab, encoder, decoder, embedding=None, hidden_size=None, embed_size=None, batch_size=None,
                 embed_dropout=0.0, params_initializer=mx.init.Uniform(0.02), abs_proj=None, prefix=None, params=None):
        super(SummarizationModel, self).__init__(prefix=prefix, params=params)
        self.encoder = encoder
        self.decoder = decoder
        self._embed = embedding
        self.batch_size = batch_size
        self.my_vocab = vocab
        print("my_vocab:", type(self.my_vocab))
        self.hidden_size = hidden_size
        print("vocab size: ", len(vocab))
        if self._embed is None:
            assert embed_size is not None, '"embed_size" cannot be None if "src_embed" is not given.'

            self._embed = nn.HybridSequential(prefix='_embed_')
            with self.name_scope():
                self._embed.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size, weight_initializer=params_initializer))
                self._embed.add(nn.Dropout(rate=embed_dropout))
        else:
            self._embed = embedding

    def encode(self, inputs, states=None):
        """inputs: (batch_size, art_sequence_length)
        """
        return self.encoder(self._embed(inputs), states)

    def decode_step(self, step_input, states):
        """One Step decoding of the Summarization Model"""
        step_output, context_vec, attention_dist, out_states, enc_states = self.decoder(self._embed(step_input), states)
        states = [out_states, enc_states]

        return step_output, states

    def decode_seq(self, inputs, states, valid_length):
        """ inputs: abs_seq (batch_size, abs_length - 1)
            states: decoder_input (cell_state, h_state), shape = (batch_size, hidden_dim)
            outputs: all V[st, ht*] + b
        """
        outputs_vs, context_vecs, attention_dists = self.decoder.decode_seq(inputs=self._embed(inputs), states=states, valid_length=valid_length)
        return outputs_vs, context_vecs, attention_dists

    def __call__(self, art, abs, states, art_valid_length=None, abs_valid_length=None):
        """ art shape = (bathc_size, art_sequence_length)
        """
        return super(SummarizationModel, self).__call__(art, abs, states, art_valid_length, abs_valid_length)

    def forward(self, art, abs, states, art_valid_length=None, abs_valid_length=None):
        """ art: (batch_size, art_length), abs: (batch_size, art_length - 1)
            encoder_outputs:
                enc_outs: (batch_size, art_sequence_length, 2*num_hidden)
                new_state: lists of new_state
        """
        encoder_outputs = self.encode(art, states)
        decoder_input = self.decoder.init_state_from_encoder(encoder_outputs, encoder_valid_length=art_valid_length)
        # outputs: all V[st, ht*] + b
        outputs, _, _ = self.decode_seq(abs, decoder_input, abs_valid_length)

        return outputs

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

class BeamSearchSummarizer(object):
    def __init__(self, model, beam_size=1, scorer=BeamSearchScorer(), max_length=100):
        self._model = model
        self._sampler = BeamSearchSampler(
            decoder=self._decode_logprob,
            beam_size=beam_size,
            eos_id=self._model.my_vocab.token_to_idx[model.my_vocab.eos_token],
            scorer=scorer,
            max_length=max_length)

    def _decode_logprob(self, step_input, states):
        out, states = self._model.decode_step(step_input, states)
        return mx.nd.log_softmax(out), states

    def summarize(self, art_seq, art_valid_length):
        batch_size = art_seq.shape[0]
        encoder_outputs = self._model.encode(art_seq, valid_length=art_valid_length)
        decoder_states = self._model.decoder.init_state_from_encoder(encoder_outputs, art_valid_length)
        inputs = mx.nd.full(shape=(batch_size,), ctx=art_seq.context, dtype=np.float32,
                            val=self._model.my_vocab.token_to_idx[self._model.my_vocab.bos_token])
        samples, scores, sample_valid_length = self._sampler(inputs, decoder_states)
        return samples, scores, sample_valid_length

class SoftmaxCrossEntropyMaskedLoss(SoftmaxCrossEntropyLoss):
    def hybrid_forward(self, F, pred, label, valid_length):
        sample_weight = F.cast(F.expand_dims(F.ones_like(label), axis=-1), dtype=np.float32)
        sample_weight = F.SequenceMask(sample_weight,
                                        sequence_length=valid_length,
                                        use_sequence_length=True,
                                        axis=1)
        return super(SoftmaxCrossEntropyMaskedLoss, self).hybrid_forward(F, pred, label, sample_weight)
