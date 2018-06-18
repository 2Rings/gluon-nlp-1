import mxnet as mx
from mxnet.base import _as_list
from mxnet.gluon import nn, rnn
from gluonnlp.model.attention_cell import MLPAttentionCell
from mxnet.gluon.block import Block, HybridBlock
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
        return self.foward(inputs, valid_length, states)

    def forward(self, inputs, valid_length = None, states = None):
        raise NotImplementedError

class SUMEncoder(S2Seq2SeqEncoder):
    def __init__(self, cell_type = 'lstm', hidden_size = 128,
                dropout = 0.0, i2h_weight_initializer = None, h2h_weight_initializer = None,
                i2h_bias_initializer = 'zeros', h2h_bias_initializer = 'zeros',
                prefix = None, params = None):
        super(SUMEncoder, self).__init__(prefix = prefix, params = params)
        self._hidden_size = hidden_size
        with self.name_scope():
            self.rnn_cells = nn.HybridSequential()
            self.rnn_cells.add(rnn.BidirectionalCell(
                l_cell= rnn.LSTMCell(hidden_size = self._hidden_size,
                i2h_weight_initializer = i2h_weight_initializer,
                h2h_weight_initializer = h2h_weight_initializer,
                i2h_bias_initializer = i2h_bias_initializer,
                h2h_bias_initializer = h2h_bias_initializer,
                prefix = 'rnn_l_'),
                r_cell= rnn.LSTMCell(hidden_size = self._hidden_size,
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
            inputs: (batch_size, sequence_length, embedding_dim)
            states: list of NDArray or None
                Initial States. The list of initial states
            return:
            outputs: (batch_size, sequence_length, num_hidden)
            new_state: lists of new_state
        """
        _, length, _ = inputs.shape

        outputs, new_state = self.rnn_cells[0].unroll(
            length = length, inputs = inputs, begin_state = None, merge_outputs = None,
            valid_length = valid_length, layout = 'NTC')
        # if valid_length is not None:
        #     outputs = mx.nd.SequenceMask(outputs, sequence_length= valid_length, use_sequence_length=True, axis = 1)

        return [outputs, new_state]

class Seq2SeqDecoder(Block):
    r"""Base class of the decoders in sequence to sequence learning models.
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
    def __init__(self, cell_type = 'lstm', num_layers = 1, num_bi_layers = 1, hidden_size = 128,
                dropout = 0.0, i2h_weight_initializer = None, h2h_weight_initializer = None,
                i2h_bias_initializer = 'zeros', h2h_bias_initializer = 'zeros',
                prefix = None, params = None):
        super(SUMEncoder, self).__init__(prefix = prefix, params = params)
        self._cell_type = _get_cell_type(cell_type)
        self._num_bi_layers = num_bi_layers
        self._hidden_size = hidden_size
        self.attention_cell = MLPAttentionCell(units = units, scaled = True, normalized=False, prefix= 'attention_')
        with self.name_scope():
            self.rnn_cells = nn.HybridSequential()
            self.rnn_cells.add(
                self._cell_type(hidden_size = self._hidden_size,
                i2h_weight_initializer = i2h_weight_initializer,
                h2h_weight_initializer = h2h_weight_initializer,
                i2h_bias_initializer = i2h_bias_initializer,
                h2h_bias_initializer = h2h_bias_initializer,
                prefix = 'dec_rnn_'
                )
            )

    def reduce_states(self, rnn_states = None):
        """
            rnn_states: [l_cell_state, l_hidden_state, r_cell_state, r_hidden_state]

        """
        l_cell_state, l_hidden_state, r_cell_state, r_hidden_state = rnn_states
        old_c = mx.nd.concat(l_cell_state, r_cell_state, axis = 1)
        old_h = mx.nd.concat(l_hidden_state, r_hidden_state, axis = 1)
        with self.name_scope():
            self._reduce_cell = nn.HybridSequential(prefix = 'reduce_cell_')
            with self._reduce_cell.name_scope():
                self._reduce_cell.add(nn.Dnese(num_hidden,
                                                activation = 'rele',
                                                weight_initializer = mx.nd.Uniform(0.1),
                                                use_bias = True,
                                                bias_initializer = mx.nd.Uniform(0.1)))

        new_c = self._reduce_cell(old_c)
        new_h = self._reduce_cell(old_h)

        return [new_c, new_h]

    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length = None):
        """encoder_outputs: list: [outputs, new_state]
        enc_states: (batch_size, length, num_hidden)
        rnn_states: [l_cell_state, l_hidden_state, r_cell_state, r_hidden_state]
        """
        #new_rnn_states: [new_c, new_h]
        enc_states, rnn_states = encoder_outputs
        new_rnn_states = self.reduce_states(rnn_states)
        decoder_input = [rnn_states, enc_states]

        return decoder_input

    def decode_seq(self, inputs, states, valid_length = None):
        #inputs: abs_seq
        #states: decoder_input
        #enc_states: (batch_size, sequence_length, num_hidden)
        length = inputs.shape[1]
        enc_states = states[1] #fixed_states
        rnn_states = states[0]
        outputs = []
        inputs = _as_list(mx.nd.split(inputs, num_outputs = length, axis = 1, squeeze_axis = True))


        for i in range(length):
            input_size = inputs[i].shape[2]
            # context vector: (batch_size, query_length, context_vec_dim) / (batch_size, num_hidden)
            # context vector: ht* = sum_i(a_i^t h_i)
            # attention_dist: (batch_size, sequence_length)
            # cell_output: (batch_size, num_hidden)
            context_vec, attention_dist = self.attention_cell(enc_states, step_input)
            cell_output, rnn_states = self.forward(inputs[i], rnn_states)
            # #V[st, ht*] + b
            # with self.name_scope():
            #     self._linear_cell = nn.HybridSequential()
            #     with self._linear_cell.name_scope()
            #         self._linear_cell.add(nn.Dense(units = output_size, weight_initializer = weight_initializer))
            #
            # output = self._linear_cell(mx.nd.concat(cell_output, context_vec, dim = 1))
            # outputs.append(output)

        return cell_output, context_vec, rnn_states, attn_dist


    def __call__(self, step_input, states):
        return self.forward(step_input, states)

    def hybrid_forward(self, step_input, states):
        #query(step_input) : (batch_size, query_length, query_dim)
        #key: (batch_size, key_length, query_dim)
        #context vector: (batch_size, query_length, context_vec_dim)
        #attention_weight: (batch_size, query_length, encode_length)
        inp_states = states[1] #fixed_states
        rnn_states = states[0]
        context_vec, attention_dist = self.attention_cell(inp_states, step_input)
        cell_output, new_rnn_states = self.rnn_cells(step_input, rnn_states)

        return cell_output, context_vec, new_rnn_states

def get_summ_encoder_decoder(cell_type = 'lstm', hidden_size = 128, dropout= 0.0,
                            i2h_weight_initializer = i2h_weight_initializer,
                            h2h_weight_initializer = h2h_weight_initializer,
                            i2h_bias_initializer = i2h_bias_initializer,
                            h2h_bias_initializer = h2h_bias_initializer,
                            prefix = 'Summarization_', params = None):
    encoder = SUMEncoder(cell_type = cell_type,
                        hidden_size = hidden_size, dropout= dropout,
                        i2h_weight_initializer = i2h_weight_initializer,
                        h2h_weight_initializer = h2h_weight_initializer,
                        i2h_bias_initializer = i2h_bias_initializer,
                        h2h_bias_initializer = h2h_bias_initializer,
                        prefix = prefix + 'enc_')

    encoder = SUMEncoder(cell_type = cell_type,
                        hidden_size = hidden_size, dropout= dropout,
                        i2h_weight_initializer = i2h_weight_initializer,
                        h2h_weight_initializer = h2h_weight_initializer,
                        i2h_bias_initializer = i2h_bias_initializer,
                        h2h_bias_initializer = h2h_bias_initializer,
                        prefix = prefix + 'dec_')

    return encoder, decoder
