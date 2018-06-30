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
            self.rnn_cells = nn.HybridSequential()
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
        # if valid_length is not None:
        #     outputs = mx.nd.SequenceMask(outputs, sequence_length= valid_length, use_sequence_length=True, axis = 1)

        return [outputs, new_state]

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
    def __init__(self, cell_type = 'lstm', num_layers = 1, num_bi_layers = 1, hidden_size = 7,
                dropout = 0.0, i2h_weight_initializer = None, h2h_weight_initializer = None,
                i2h_bias_initializer = 'zeros', h2h_bias_initializer = 'zeros',
                prefix = None, params = None):
        super(SUMDecoder, self).__init__(prefix = prefix, params = params)
        self._cell_type = rnn.LSTMCell
        self._num_bi_layers = num_bi_layers
        self._hidden_size = hidden_size
        self.attention_cell = MLPAttentionCell(units = self._hidden_size, normalized=False, prefix= 'attention_')
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
            self._reduce_cell = nn.HybridSequential(prefix = 'reduce_cell_')
            with self._reduce_cell.name_scope():
                self._reduce_cell.add(nn.Dense(self._hidden_size,
                                                activation = 'relu',
                                                weight_initializer = mx.init.Uniform(0.1),
                                                use_bias = True,
                                                bias_initializer = mx.init.Uniform(0.1)))

        with self.name_scope():
            self.abs_proj = nn.HybridSequential(prefix = 'abs_proj_')
            with self.abs_proj.name_scope():
                self.abs_proj.add(nn.Dense(self._hidden_size,
                                    # activation = 'relu',
                                    weight_initializer = mx.init.Uniform(0.1),
                                    use_bias = True,
                                    flatten=False,
                                    bias_initializer = mx.init.Uniform(0.1)))

        with self.name_scope():
            self._linear_layer = nn.HybridSequential()
            with self._linear_layer.name_scope():
                self._linear_layer.add(nn.Dense(960, weight_initializer = mx.init.Uniform(0.1)))



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
        old_c = mx.nd.concat(l_cell_state, r_cell_state, dim = 1)
        old_h = mx.nd.concat(l_hidden_state, r_hidden_state, dim = 1)


        new_c = self._reduce_cell(old_c)
        new_h = self._reduce_cell(old_h)

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
            output_vs, context_vec, attention_dist, rnn_states = self.forward(inputs[i], rnn_states, enc_states)
            # #V'(V[st, ht*] + b) + b'
            outputs.append(output_vs)
            context_vec = mx.ndarray.reshape(context_vec, shape = (batch_size, -1))
            attention_dist = mx.ndarray.reshape(attention_dist, shape = (batch_size, -1))
            context_vecs.append(context_vec)
            attention_dists.append(attention_dist)

        return outputs, context_vecs, attention_dists


    def __call__(self, step_input, states):
        return self.forward(step_input, states)

    def forward(self, step_input, rnn_states, enc_states):
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

        cell_output, new_rnn_states = self._rnn_cells[0](step_input, rnn_states)
        att_cell_out = mx.ndarray.expand_dims(cell_output, axis = 1)
        context_vec, attention_dist = self.attention_cell(att_cell_out,enc_states)
        output_linear = self.abs_proj[0](mx.nd.concat(att_cell_out, context_vec, dim = 2))
        output_vs = self._linear_layer[0](output_linear)

        return output_vs, context_vec, attention_dist, new_rnn_states

def get_summ_encoder_decoder(cell_type = 'lstm', hidden_size = 128, dropout= 0.0,
                            i2h_weight_initializer = None,
                            h2h_weight_initializer = None,
                            i2h_bias_initializer = 'zeros',
                            h2h_bias_initializer = 'zeros',
                            prefix = 'Summarization_', params = None):
    encoder = SUMEncoder(cell_type = cell_type,
                        hidden_size = hidden_size, dropout= dropout,
                        i2h_weight_initializer = i2h_weight_initializer,
                        h2h_weight_initializer = h2h_weight_initializer,
                        i2h_bias_initializer = i2h_bias_initializer,
                        h2h_bias_initializer = h2h_bias_initializer,
                        prefix = prefix + 'enc_')

    decoder = SUMDecoder(cell_type = cell_type,
                        hidden_size = hidden_size, dropout= dropout,
                        i2h_weight_initializer = i2h_weight_initializer,
                        h2h_weight_initializer = h2h_weight_initializer,
                        i2h_bias_initializer = i2h_bias_initializer,
                        h2h_bias_initializer = h2h_bias_initializer,
                        prefix = prefix + 'dec_')

    return encoder, decoder
