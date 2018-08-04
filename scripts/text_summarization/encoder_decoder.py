import mxnet as mx
from functools import partial
from mxnet.base import _as_list
from mxnet.gluon import nn, rnn
from gluonnlp.model.attention_cell import MLPAttentionCell
from mxnet.gluon.block import Block, HybridBlock


__all__ = ['Seq2SeqEncoder', 'Seq2Seq2SeqDecoder', 'SUMEncoder', 'SUMDecoder', 'get_summ_encoder_decoder']
class Seq2SeqEncoder(Block):
    """Base Class of the encoders in sequence to sequence learning models."""

    def __call__(self, inputs, valid_length = None, states = None):
        """Encode the input sequence.
        Parameters
        ----------
        inputs : NDArray
            The input sequence, Shape (batch_size, sequence_length, embedding_dim).
        valid_length : NDArray or None, default None
            The valid length of the input sequence, Shape (batch_size,). This is used when the
            input sequences are padded. If set to None, all elements in the sequence are used.
        states : list of NDArrays or None, default None
            List that contains the initial states of the encoder.
        Returns
        -------
        outputs : list
            Outputs of the encoder.
        """
        # return self.foward(inputs, valid_length, states)
        return super(Seq2SeqEncoder, self).__call__(inputs, valid_length, states)

    def forward(self, inputs, valid_length = None, states = None):
        raise NotImplementedError

class SUMEncoder(Seq2SeqEncoder):
    def __init__(self, cell_type = 'lstm', hidden_size = 128,
                dropout = 0.0, i2h_weight_initializer = None, h2h_weight_initializer = None,
                i2h_bias_initializer = 'zeros', h2h_bias_initializer = 'zeros',
                prefix = None, params = None):
        super(SUMEncoder, self).__init__(prefix = prefix, params = params)
        self.hidden_size = hidden_size
        with self.name_scope():
            self.rnn_cells = rnn.SequentialRNNCell()
            self.rnn_cells.add(rnn.BidirectionalCell(
                l_cell= rnn.LSTMCell(hidden_size = self.hidden_size,
                i2h_weight_initializer = i2h_weight_initializer,
                h2h_weight_initializer = h2h_weight_initializer,
                i2h_bias_initializer = i2h_bias_initializer,
                h2h_bias_initializer = h2h_bias_initializer,
                prefix = 'rnn_l_'),
                r_cell= rnn.LSTMCell(hidden_size = self.hidden_size,
                i2h_weight_initializer = i2h_weight_initializer,
                h2h_weight_initializer = h2h_weight_initializer,
                i2h_bias_initializer = i2h_bias_initializer,
                h2h_bias_initializer = h2h_bias_initializer,
                prefix = 'rnn_r_')))

    def __call__(self, inputs, states = None, valid_length = None):
        """Parameters:
        #inputs: (batch_size, sequence_length, embedding_dim) NDArray
        #states: list of NDArray or None
        # valid_length: (batch_size,)
        #Return:
        #encoder_outputs: list
        #   Outputs of the encoder"""
        return self.forward(inputs, states, valid_length)

    def forward(self, inputs, states = None, valid_length = None):
        """
            inputs: (batch_size, art_sequence_length, embedding_dim)
            states: list of NDArray or None
                Initial States. The list of initial states
            return:
            outputs: (batch_size, sequence_length, 2*num_hidden)
            new_state: lists of new_state
        """
        _, length, _ = inputs.shape

        outputs, new_state = self.rnn_cells[0].unroll(
            length = length, inputs = inputs, begin_state = states, merge_outputs = True,
            valid_length = valid_length, layout = 'NTC')

        return [outputs, new_state]

class Attention(HybridBlock):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        #attention
        with self.name_scope():
            self.encoder_features = nn.HybridSequential(prefix='encoder_features')
            with self.encoder_features.name_scope():
                self.encoder_features.add(nn.Conv2D(self.hidden_size*2, [1,1], layout='NHWC', weight_initializer = mx.init.Uniform(0.02), use_bias=False))

            self.attention_v = nn.HybridSequential(prefix='attention_v')
            with self.attention_v.name_scope():
                self.attention_v.add(nn.Dropout(0.0))

            self.decoder_featrures = nn.HybridSequential(prefix='decoder_features')
            with self.decoder_featrures.name_scope():
                self.decoder_featrures.add(nn.Dense(self.hidden_size*2, weight_initializer=mx.init.Uniform(0.02), use_bias=False))

            self.attention_tanh = nn.HybridSequential(prefix='attention_tanh')
            with self.attention_tanh.name_scope():
                self.attention_tanh.add(nn.Dense(self.hidden_size,
                                                 activation='tanh',
                                                 weight_initializer=mx.init.Uniform(0.02),
                                                 use_bias=True,
                                                 bias_initializer=mx.init.Uniform(0.02)))

    def __call__(self, encoder_states, decoder_state):
        return self.forward(encoder_states, decoder_state)

    def masked_attention(self, e, enc_padding_mask):
        attn_dist = mx.ndarray.softmax(e)
        attn_dist *= enc_padding_mask
        masked_sums = mx.ndarray.sum(attn_dist, axis=1)
        return attn_dist / mx.ndarray.reshape(masked_sums, [-1, 1])

    def forward(self, encoder_states, decoder_state):
        batch_size = encoder_states.shape[0]
        encoder_states = mx.ndarray.expand_dims(encoder_states, axis=2)
        encoder_feature = self.encoder_features[0](encoder_states)
        decoder_feature = self.decoder_featrures[0](decoder_state)
        decoder_feature = mx.ndarray.expand_dims(mx.ndarray.expand_dims(decoder_feature, 1), 1)
        feature = self.attention_tanh[0](encoder_feature + decoder_feature)
        e = mx.ndarray.sum(self.attention_v[0](feature), axis=[2,3])
        attn_dist = self.masked_attention(e)

        context_vector = mx.ndarray.sum(mx.ndarray.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [1, 2])
        context_vector = mx.ndarray.reshape(context_vector, [-1, attn_dist])

        return context_vector, attn_dist



class Seq2SeqDecoder(Block):
    """Base class of the decoders in sequence to sequence learning models.
    In the forward function, it generates the one-step-ahead decoding output.
    """
    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length = None):
        raise NotImplementedError

    def decode_seq(self, inputs, states, valid_length = None):
        raise NotImplementedError

    def __call__(self, step_input, states):
        return self.forward(step_input, states)

    def forward(self, step_input, states):
        raise NotImplementedError

class SUMDecoder(Seq2SeqDecoder):
    def __init__(self, cell_type = 'lstm', num_layers = 1, num_bi_layers = 1, hidden_size = 7, embedding_size = 64, vocab = None,
                dropout = 0.0, i2h_weight_initializer = None, h2h_weight_initializer = None,
                i2h_bias_initializer = 'zeros', h2h_bias_initializer = 'zeros',
                prefix = None, params = None):
        super(SUMDecoder, self).__init__(prefix = prefix, params = params)
        self._cell_type = rnn.LSTMCell
        self._num_bi_layers = num_bi_layers
        self._hidden_size = hidden_size
        self.vocab = vocab
        self._embedding_size = embedding_size

        print("vocab_lenght: ", len(self.vocab))
        self.attention_cell = MLPAttentionCell(units=2*self._hidden_size, normalized=False, prefix= 'attention_')
        with self.name_scope():
            self._rnn_cells = nn.HybridSequential()
            self._rnn_cells.add(
                self._cell_type(hidden_size = self._hidden_size,
                i2h_weight_initializer = i2h_weight_initializer,
                h2h_weight_initializer = h2h_weight_initializer,
                i2h_bias_initializer = i2h_bias_initializer,
                h2h_bias_initializer = h2h_bias_initializer,
                prefix = 'dec_rnn_'
                )
            )

        with self.name_scope():
            self._reduce_cell_c = nn.HybridSequential(prefix = 'reduce_cell_c')
            with self._reduce_cell_c.name_scope():
                self._reduce_cell_c.add(nn.Dense(self._hidden_size,
                                                activation = 'relu',
                                                weight_initializer = mx.init.Uniform(0.02),
                                                use_bias = True,
                                                bias_initializer = mx.init.Uniform(0.02)))

        with self.name_scope():
            self._reduce_cell_h = nn.HybridSequential(prefix = 'reduce_cell_h')
            with self._reduce_cell_h.name_scope():
                self._reduce_cell_h.add(nn.Dense(self._hidden_size,
                                                activation = 'relu',
                                                weight_initializer = mx.init.Uniform(0.02),
                                                use_bias = True,
                                                bias_initializer = mx.init.Uniform(0.02)))

        with self.name_scope():
            self.abs_proj = nn.HybridSequential(prefix = 'abs_proj_')
            with self.abs_proj.name_scope():
                self.abs_proj.add(nn.Dense(self._hidden_size,
                                    # activation = 'relu',
                                    weight_initializer = mx.init.Uniform(0.02),
                                    use_bias = True,
                                    flatten=False,
                                    bias_initializer = mx.init.Uniform(0.02)))

        with self.name_scope():
            self._linear_layer = nn.HybridSequential()
            with self._linear_layer.name_scope():
                self._linear_layer.add(nn.Dense(len(self.vocab), weight_initializer = mx.init.Uniform(0.02)))

        with self.name_scope():
            self.dec_linear = nn.HybridSequential(prefix='dec_linear')
            with self.dec_linear.name_scope():
                self.dec_linear.add(nn.Dense(self._embedding_size, weight_initializer = mx.init.Uniform(0.02)))



    def reduce_states(self, rnn_states = None):
        """
            rnn_states: [l_cell_state, l_hidden_state, r_cell_state, r_hidden_state], shape(batch_size, hidden_dim)
            return: list of new_states, shape = [(batch_size, hidden_dim), (batch_size, hidden_dim)]
            rnn_states: [l_cell_state, l_hidden_state, r_cell_state, r_hidden_state],
            shape = (batch_size, hidden_dim)
            returns:
                list: cell_state and hidden_state
        """
        l_cell_state, l_hidden_state, r_cell_state, r_hidden_state = rnn_states
        old_c = mx.nd.concat(l_cell_state, r_cell_state, dim=1)
        old_h = mx.nd.concat(l_hidden_state, r_hidden_state, dim=1)
        new_c = self._reduce_cell_c[0](old_c)
        new_h = self._reduce_cell_h[0](old_h)

        return [new_c, new_h]

    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length = None):
        """encoder_outputs: list: [outputs, new_state]
            enc_states: (batch_size, length, num_hidden)
            rnn_states: [l_cell_state, l_hidden_state, r_cell_state, r_hidden_state]
            enc_states: shape = (batch_size, art_sequence_length, 2*hidden_dim)
            new_rnn_states: [new_c, new_h]
            decoder_input = [new_rnn_states, enc_states]
        """
        enc_states, rnn_states = encoder_outputs
        new_rnn_states = self.reduce_states(rnn_states)

        return [new_rnn_states, enc_states]

    def decode_seq(self, inputs, states, valid_length = None):
        """#inputs: abs_seq
        #states: decoder_input
        #enc_states: (batch_size, sequence_length, 2 * num_hidden)
        returns : list of cell_output(batch_size, hidden_dim), context_vec(batch_size, 2*hidden_dim), attention_dist(batch_size, art_sequence_length)"""
        """ inputs: abs_seq (batch_size, abs_length - 1, embedding_dim)
            states: decoder_input
            enc_states: (batch_size, sequence_length, 2*num_hidden)
            returns:
                list of cell_output, context_vec(batch_size, 2* hidden_dim), correponding attention_dist(batch_size, art_sequence_length)
        """
        length = inputs.shape[1]
        batch_size = inputs.shape[0]

        enc_states = states[1] #fixed_states
        rnn_states = states[0]
        outputs = []
        context_vecs = []
        attention_dists = []
        inputs = _as_list(mx.nd.split(inputs, num_outputs = length, axis = 1, squeeze_axis = True))

        for i in range(length):
            output_vs, context_vec, attention_dist, rnn_states, _ = self.forward(inputs[i], rnn_states, enc_states)
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

    def forward(self, step_input, rnn_states, enc_states = None):
        """
        Input: step_input: shape = (batch_size, embed_dim)
        states: list of state (batch_size, hidden_dim)
        """
        """query(step_input) : (batch_size, query_length, query_dim)
        #key: (batch_size, key_length, query_dim)
        #context vector: (batch_size, query_length, context_vec_dim)
        #attention_weight: (batch_size, query_length, encode_length)
        step_input: (batch_size, embedding_dim)
        states: list(cell_state, h_state) (batch_size, hidden_dim)
        returns:
            new_states: list(cell_state, h_state) (batch_size, hidden_dim)
        """

        dec_state = mx.nd.concat(rnn_states[0], rnn_states[1], dim=1)
        batch_size = step_input.shape[0]
        # print("dec_state: ", dec_state.shape)
        dec_state = mx.ndarray.expand_dims(dec_state, axis=1)
        context_vec, attention_dist = self.attention_cell(dec_state, enc_states)
        context_vec_inp = mx.ndarray.reshape(context_vec, shape = (batch_size, -1))
        step_input = mx.nd.concat(step_input, context_vec_inp, dim=1)
        step_input = self.dec_linear[0](step_input)
        cell_output, new_rnn_states = self._rnn_cells[0](step_input, rnn_states)
        attn_cell_out = mx.ndarray.expand_dims(cell_output, axis=1)
        output_linear = self.abs_proj[0](mx.nd.concat(attn_cell_out, context_vec, dim=2))
        output_vs = self._linear_layer[0](output_linear)

        return output_vs, context_vec, attention_dist, new_rnn_states, enc_states


def get_summ_encoder_decoder(cell_type='lstm', hidden_size=128, embedding_size = 64,
                             vocab = None, dropout=0.0,
                            i2h_weight_initializer=None,
                            h2h_weight_initializer=None,
                            i2h_bias_initializer='zeros',
                            h2h_bias_initializer='zeros',
                            prefix='Summarization_', params = None):
    encoder = SUMEncoder(cell_type = cell_type,
                        hidden_size = hidden_size, dropout= dropout,
                        i2h_weight_initializer = i2h_weight_initializer,
                        h2h_weight_initializer = h2h_weight_initializer,
                        i2h_bias_initializer = i2h_bias_initializer,
                        h2h_bias_initializer = h2h_bias_initializer,
                        prefix = prefix + 'enc_')

    decoder = SUMDecoder(cell_type = cell_type,
                        hidden_size = hidden_size,
                         embedding_size= embedding_size,
                         vocab = vocab, dropout= dropout,
                        i2h_weight_initializer = i2h_weight_initializer,
                        h2h_weight_initializer = h2h_weight_initializer,
                        i2h_bias_initializer = i2h_bias_initializer,
                        h2h_bias_initializer = h2h_bias_initializer,
                        prefix = prefix + 'dec_')

    return encoder, decoder
