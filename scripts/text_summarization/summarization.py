import numpy as np
from mxnet.gluon import Block
from mxnet.gluon import nn
import nxnet as mx

__all__ = ['SummarizationModel']


class SummarizationModel(Block):

    def __init__(self, vocab, encoder, decoder, embedding = None, embed_size = None, embed_dropout = 0.0, embed_initializer = mx.init.Uniform(0.1), abs_proj = None, prefix = None, params = None):
        super(SummarizationModel, self).__init__(prefix = prefix, params)
        self.vocab = vocab
        self.encoder = encoder
        self.decoder = decoder

        if embed is None:
            assert embed_size is not None, '"embed_size" cannot be None if "src_embed" is not given.'

            with self.name_scope():
                self._embed = nn.HybridSequential(prefix = '_embed_')
                with self._embed.name_scope():
                    self._embed.add(nn.Embedding(input_dim = len(vocab), output_dim = embed_size, weight_initializer = embed_initializer))
                    self._embed.add(nn.Dropout(rate = embed_dropout))

        else:
            self._embed = embed

        if abs_proj is None:
            with self.name_scope():
                self.abs_proj = nn.Dense(units = len(vocab), flatten = False, prefix = 'abs_proj_')
        else:
            self.abs_proj = abs_proj


    def encode(self, inputs, states = None, valid_length = None):
        #inputs: (batch_size, sequence_length)
        return self.encoder(self._embed(inputs), states, valid_length)

    def decode_step(self, step_input, states, valid_length = None):

        """One Step decoding of the Summarization Model"""
        step_output, context_vec, states = self.decoder(self._embed(step_input), states)
        output = self.abs_proj(mx.nd.concat(step_output, context_vec, dim = 1))

        return step_output, states

    def decode_seq(self, inputs, states):
        #inputs: abs_seq
        #states: decoder_input
        #outputs: all V[st, ht*] + b
        cell_output, context_vec, states, attention_dist = self.decoder.decoder_seq(inputs = self._embed(inputs), states = states, valid_length = valid_length)
        # outputs = self.abs_proj(outputs)
        outputs = self.abs_proj(mx.nd.concat(cell_output, context_vec, dim = 1))
        return outputs, states, attention_dist


    def __call__(self, art, abs, art_valid_length = None, abs_valid_length = None):
        return super(SummarizationModel, self).__call__(art, abs, art_valid_length, abs_valid_length)

    def forward(self, art, abs, art_valid_length = None, abs_valid_length = None):
        encoder_outputs = self.encode(art, valid_length = art_valid_length)
        decoder_input = self.decoder.init_state_from_encoder(encoder_outputs, encoder_valid_length = art_valid_length)
        #outputs: all V[st, ht*] + b
        outputs, _, _ = self.decode_seq(abs, decoder_input, abs_valid_length)

        return outputs
