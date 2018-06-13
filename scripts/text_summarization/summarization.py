__all__ = ['SummarizationModel']

import numpy as np
from mxnet.gluon import Block
from mxnet.gluon import nn
import nxnet as mx

class SummarizationModel(Block):

    def __init__(self, vocab, encoder, decoder, embed_size = None, embed_dropout = 0.0, embed_initializer = mx.init.Uniform(0.1), abs_proj = None, prefix = None, params = None):
        super(SummarizationModel, self).__init__(prefix = prefix, params, params)
        self.vocab = vocab
        self.encoder = encoder
        self.decoder = decoder

        if embed_dropout = None:
            embed_dropout = 0.0

    def encode(self, inputs, states = None, valid_length = None):
        return self.encoder(self.src_embed(inputs), states, valid_length)

    def decode_seq(self, inputs, states, valid_length = None):
        pass

    def decode_step(self, step_input, states):
        """One Step decoding of the Summarization Model"""
        pass


    def __call__(self, art, abs, art_valid_length = None, abs_valid_length = None):
        return super(SummarizationModel, self).__call__(art, abs, art_valid_length, abs_valid_length)

    def forward(self, art, abs, art_valid_length = None, abs_valid_length = None):
        encoder_outputs = self.encode(art, valid_length = art_valid_length)
        decoder_states = self.decoder.init_state_from_encoder(encoder_outputs, encoder_valid_length = art_valid_length)

        outputs, _, additional_outputs = self.decode_step(abs, decoder_states, abs_valid_length)

        return outputs, additional_outputs
