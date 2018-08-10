import mxnet as mx
from functools import partial
from mxnet.base import _as_list
from mxnet.gluon import nn, rnn
from gluonnlp.model.attention_cell import MLPAttentionCell
from mxnet.gluon.block import Block, HybridBlock


__all__ = ['Pointer_Generator_Encoder', 'Pointer_Generator_Decoder', 'get_summ_encoder_decoder']
class Pointer_Generator_Encoder(Block):
    def __init__(self, hidden_size = 128,
                dropout = 0.0, i2h_weight_initializer = None, h2h_weight_initializer = None,
                i2h_bias_initializer = 'zeros', h2h_bias_initializer = 'zeros',
                prefix = None, params = None):
        super(Pointer_Generator_Encoder, self).__init__(prefix = prefix, params = params)
        self.hidden_size = hidden_size
        # self.bilstm = nn.HybridSequential(prefix='enc_bilstm')
        with self.name_scope():
            self.bilstm = rnn.LSTM(self.hidden_size,
                                 bidirectional=True,
                                 layout='NTC',
                                 i2h_weight_initializer=i2h_weight_initializer,
                                 h2h_weight_initializer=h2h_weight_initializer,
                                 i2h_bias_initializer=i2h_bias_initializer,
                                 h2h_bias_initializer=h2h_bias_initializer)

    def __call__(self, inputs, states = None):
        """Parameters:
        #inputs: (batch_size, sequence_length, embedding_dim) NDArray
        #states: list of NDArray or None
        # valid_length: (batch_size,)
        #Return:
        #encoder_outputs: list
        #   Outputs of the encoder"""
        return self.forward(inputs, states)

    def forward(self, inputs, states = None):
        """
            inputs: (batch_size, art_sequence_length, embedding_dim)
            states: list of NDArray or None
                Initial States. The list of initial states
            return:
            outputs: (batch_size, sequence_length, 2*num_hidden)
            new_state: lists of new_state
        """
        out, new_state = self.bilstm(inputs, states)

        return [out, new_state]

    def begin_state(self, *args, **kwargs):
        return self.bilstm.begin_state(*args, **kwargs)

class Pointer_Generator_Decoder(Block):
    def __init__(self, cell_type = 'lstm', num_layers = 1, num_bi_layers = 1, hidden_size = 7, embedding_size = 64, vocab = None,
                dropout = 0.0, i2h_weight_initializer = None, h2h_weight_initializer = None,
                i2h_bias_initializer = 'zeros', h2h_bias_initializer = 'zeros',
                prefix = None, params = None):
        super(Pointer_Generator_Decoder, self).__init__(prefix = prefix, params = params)
        self._cell_type = rnn.LSTMCell
        self._num_bi_layers = num_bi_layers
        self._hidden_size = hidden_size
        self.vocab = vocab
        self._embedding_size = embedding_size

        self._reduce_cell_c = nn.HybridSequential(prefix='reduce_cell_c')
        self._reduce_cell_h = nn.HybridSequential(prefix='reduce_cell_h')

        self.dec_out_linear = nn.HybridSequential(prefix='dec_out_linear')
        self.abs_proj = nn.HybridSequential(prefix='abs_proj_')
        self._linear_layer = nn.HybridSequential(prefix='linear_layer_')
        self.dec_linear = nn.HybridSequential(prefix='dec_linear_')

        self.dec_lstm = nn.HybridSequential('dec_lstm_')
        with self.name_scope():
            self.dec_lstm.add(rnn.LSTMCell(hidden_size = self._hidden_size,
                                        i2h_weight_initializer = i2h_weight_initializer,
                                        h2h_weight_initializer = h2h_weight_initializer,
                                        i2h_bias_initializer = i2h_bias_initializer,
                                        h2h_bias_initializer = h2h_bias_initializer))

            self.attention_cell = MLPAttentionCell(units=2 * self._hidden_size, normalized=False, prefix='attention_')

            self._reduce_cell_c.add(nn.Dense(self._hidden_size,
                                            activation = 'relu',
                                            weight_initializer = mx.init.Uniform(0.02),
                                            use_bias = True,
                                            bias_initializer = mx.init.Uniform(0.02)))

            self._reduce_cell_h.add(nn.Dense(self._hidden_size,
                                            activation = 'relu',
                                            weight_initializer = mx.init.Uniform(0.02),
                                            use_bias = True,
                                            bias_initializer = mx.init.Uniform(0.02)))

            self.dec_out_linear.add(nn.Dense(2*self._hidden_size,
                                             weight_initializer=mx.init.Uniform(0.02),
                                             use_bias=True,
                                             bias_initializer=mx.init.Uniform(0.02)))

            self.abs_proj.add(nn.Dense(self._hidden_size,
                                            weight_initializer = mx.init.Uniform(0.02),
                                            use_bias = True,
                                            flatten=False,
                                            bias_initializer = mx.init.Uniform(0.02)))

            self._linear_layer.add(nn.Dense(len(self.vocab), weight_initializer = mx.init.Uniform(0.02)))

            self.dec_linear.add(nn.Dense(self._embedding_size, weight_initializer = mx.init.Uniform(0.02)))

    def reduce_states(self, rnn_states = None):
        l_state, r_state = rnn_states
        old_h = mx.nd.concat(l_state[0], r_state[0], dim=1)
        old_c = mx.nd.concat(l_state[1], r_state[1], dim=1)
        new_c = self._reduce_cell_c[0](old_c)
        new_h = self._reduce_cell_h[0](old_h)
        return [new_h, new_c]

    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length = None):
        enc_states, rnn_states = encoder_outputs
        new_rnn_states = self.reduce_states(rnn_states)

        return [new_rnn_states, enc_states]

    def decode_seq(self, inputs, states, valid_length = None):
        length = inputs.shape[1]
        batch_size = inputs.shape[0]

        enc_states = states[1]
        # print("enc_states: ", enc_states.shape)
        rnn_states = states[0]
        outputs = []
        context_vecs = []
        attention_dists = []
        inputs = _as_list(mx.nd.split(inputs, num_outputs = length, axis = 1, squeeze_axis = True))
        max_enc_outputs = mx.ndarray.max(enc_states, axis=1)

        for i in range(length):
            output_vs, context_vec, attention_dist, rnn_states, _ = self.forward(inputs[i], rnn_states, enc_states, max_enc_outputs)
            # #V'(V[st, ht*] + b) + b'
            outputs.append(output_vs)
            context_vec = mx.ndarray.reshape(context_vec, shape = (batch_size, -1))
            attention_dist = mx.ndarray.reshape(attention_dist, shape = (batch_size, -1))
            context_vecs.append(context_vec)
            attention_dists.append(attention_dist)

        return outputs, context_vecs, attention_dists


    def __call__(self, step_input, states):
        rnn_states = states[0]
        enc_states = states[1]
        return self.forward(step_input, rnn_states, enc_states)

    def forward(self, step_input, rnn_states, enc_states=None, max_enc_outputs=None):
        # attention_decoder
        dec_state = mx.nd.concat(rnn_states[0], rnn_states[1], dim=1)
        dec_state = self.dec_out_linear[0](dec_state)
        batch_size = step_input.shape[0]
        dec_state = mx.ndarray.expand_dims(dec_state, axis=1)
        context_vec, attention_dist = self.attention_cell(dec_state, enc_states)
        context_vec_inp = mx.ndarray.reshape(context_vec, shape=(batch_size, -1))
        step_input = mx.nd.concat(step_input, context_vec_inp, dim=1)
        step_input = self.dec_linear[0](step_input)
        cell_output, new_rnn_states = self.dec_lstm[0](step_input, rnn_states)
        attn_cell_out = mx.ndarray.expand_dims(cell_output, axis=1)
        output_linear = self.abs_proj[0](mx.nd.concat(attn_cell_out, context_vec, dim=2))
        output_vs = self._linear_layer[0](output_linear)

        # decoder
        # cell_output, new_rnn_states = self.dec_lstm[0](step_input, rnn_states)
        # output_vs = self.abs_proj[0](cell_output)
        # output_vs = self._linear_layer[0](output_vs)
        # context_vec = None
        # attention_dist = None

        return output_vs, context_vec, attention_dist, new_rnn_states, enc_states


def get_summ_encoder_decoder(cell_type='lstm', hidden_size=128, embedding_size = 64,
                             vocab = None, dropout=0.0,
                            i2h_weight_initializer=mx.init.Uniform(0.02),
                            h2h_weight_initializer=mx.init.Uniform(0.02),
                            i2h_bias_initializer='zeros',
                            h2h_bias_initializer='zeros',
                            prefix='Summarization_', params = None):
    encoder = Pointer_Generator_Encoder(hidden_size = hidden_size, dropout= dropout,
                        i2h_weight_initializer = i2h_weight_initializer,
                        h2h_weight_initializer = h2h_weight_initializer,
                        i2h_bias_initializer = i2h_bias_initializer,
                        h2h_bias_initializer = h2h_bias_initializer,
                        prefix = prefix + 'enc_')

    decoder = Pointer_Generator_Decoder(cell_type = cell_type,
                        hidden_size = hidden_size,
                         embedding_size= embedding_size,
                         vocab = vocab, dropout= dropout,
                        i2h_weight_initializer = i2h_weight_initializer,
                        h2h_weight_initializer = h2h_weight_initializer,
                        i2h_bias_initializer = i2h_bias_initializer,
                        h2h_bias_initializer = h2h_bias_initializer,
                        prefix = prefix + 'dec_')
    return encoder, decoder
