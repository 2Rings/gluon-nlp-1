import numpy as np
from mxnet.gluon import Block
from mxnet.gluon import nn
import mxnet as mx

__all__ = ['SummarizationModel']


class SummarizationModel(Block):

    def __init__(self, vocab, encoder, decoder, embedding = None, hidden_dim = None, embed_size = None, embed_dropout = 0.0, embed_initializer = mx.init.Uniform(0.1), abs_proj = None, prefix = None, params = None):
        super(SummarizationModel, self).__init__(prefix = prefix, params = params)
<<<<<<< HEAD
=======
        print "#51: model.__init__"
>>>>>>> c727e446161eb1da907678319584c94d8dee0c58
        self.vocab = vocab
        # print "SummarizationModel.len(vocab): ", len(vocab)
        self.encoder = encoder
        self.decoder = decoder
        self._embed = embedding
        if self._embed is None:
            assert embed_size is not None, '"embed_size" cannot be None if "src_embed" is not given.'

            with self.name_scope():
                self._embed = nn.HybridSequential(prefix = '_embed_')
                with self._embed.name_scope():
<<<<<<< HEAD
                    print "embediding"
=======
                    print "#52: embediding"
>>>>>>> c727e446161eb1da907678319584c94d8dee0c58
                    self._embed.add(nn.Embedding(input_dim = len(vocab), output_dim = embed_size, weight_initializer = embed_initializer))
                    self._embed.add(nn.Dropout(rate = embed_dropout))

        else:
            self._embed = embed

        if abs_proj is None:
            #linear layer for concatenate cell_out and context_vec: V[st, ht*] + b
            print "#53: abs_proj"
            with self.name_scope():
                self.abs_proj = nn.Dense(units = 2*hidden_dim, flatten = False, prefix = 'abs_proj_')
        else:
            self.abs_proj = abs_proj


    def encode(self, inputs, states = None, valid_length = None):
<<<<<<< HEAD
        """inputs: (batch_size, art_sequence_length)"""
=======
        """inputs: (batch_size, art_sequence_length)
        """

>>>>>>> c727e446161eb1da907678319584c94d8dee0c58
        return self.encoder(self._embed(inputs), states, valid_length)

    def decode_step(self, step_input, states, valid_length = None):

        """One Step decoding of the Summarization Model"""
        print "summarization_test.decode_step: ", step_input.shape, states.shape
        step_output, context_vec, states = self.decoder(self._embed(step_input), states)
        output = self.abs_proj(mx.nd.concat(step_output, context_vec, dim = 1))

        return step_output, states

    def decode_seq(self, inputs, states, valid_length):
<<<<<<< HEAD
        """inputs: abs_seq : shape = (batch_size, abs_sequence_length - 1)
        #states: decoder_input
        #outputs: all V[st, ht*] + b"""
        cell_outputs, context_vecs, attention_dists = self.decoder.decode_seq(inputs = self._embed(inputs), states = states, valid_length = valid_length)
        #outputs: all V[st, ht*] + b (list)
        outputs = [self.abs_proj(mx.nd.concat(cell_output, context_vec, dim = 1)) for cell_output, context_vec in zip(cell_outputs, context_vecs)]
=======
        """ inputs: abs_seq (batch_size, abs_length - 1)
            states: decoder_input (cell_state, h_state), shape = (batch_size, hidden_dim)
            outputs: all V[st, ht*] + b
        """
        print "#64"
        cell_outputs, context_vecs, attention_dists = self.decoder.decode_seq(inputs = self._embed(inputs), states = states, valid_length = valid_length)
        outputs = [self.abs_proj(mx.nd.concat(cell_output, context_vec, dim = 1)) for cell_output, context_vec in zip(cell_outputs, context_vecs)]
        print "outputs: type", type(outputs[0]), type(cell_outputs[0])
>>>>>>> c727e446161eb1da907678319584c94d8dee0c58
        return outputs, context_vecs, attention_dists


    def __call__(self, art, abs, art_valid_length = None, abs_valid_length = None):
<<<<<<< HEAD
        """ art shape = (bathc_size, art_sequence_length)
        """
        return super(SummarizationModel, self).__call__(art, abs, art_valid_length, abs_valid_length)

    def forward(self, art, abs, art_valid_length = None, abs_valid_length = None):
        """
            art shape = (batch_size, art_sequence_length)
            abs shape = (batch_size, abs_sequence_length - 1)
            list of states: [(batch_size, hidden_dim), (batch_size, hidden_dim)]
            encoder_outputs = (batch_size, art_sequence_length, 2* hidden_dim)
        """
=======
        print "#60 model.__call__"
        return super(SummarizationModel, self).__call__(art, abs, art_valid_length, abs_valid_length)

    def forward(self, art, abs, art_valid_length = None, abs_valid_length = None):
        """ art: (batch_size, art_length), abs: (batch_size, art_length - 1)
            encoder_outputs:
                outputs: (batch_size, art_sequence_length, 2*num_hidden)
                new_state: lists of new_state
        """
        print "#61: "
>>>>>>> c727e446161eb1da907678319584c94d8dee0c58
        encoder_outputs = self.encode(art, valid_length = art_valid_length)

        decoder_input = self.decoder.init_state_from_encoder(encoder_outputs, encoder_valid_length = art_valid_length)

        #outputs: all V[st, ht*] + b
        outputs, _, _ = self.decode_seq(abs, decoder_input, abs_valid_length)

        return outputs
