import mxnet as mx
from mxnet.gluon import nn, rnn
from gluonnlp.model.attention_cell import MLPAttentionCell

class Seq2SeqEncoder(Block):
    pass

class SUMEncoder(S2Seq2SeqEncoder):
    def __init__(self, cell_type = 'lstm', num_layers = 1, num_bi_layers = 1, hidden_size = 128,
                dropout = 0.0, i2h_weight_initializer = None, h2h_weight_initializer = None,
                i2h_bias_initializer = 'zeros', h2h_bias_initializer = 'zeros',
                prefix = None, params = None):
        super(SUMEncoder, self).__init__(prefix = prefix, params = params)
        self._cell_type = _get_cell_type(cell_type)
        self._num_bi_layers = num_bi_layers
        self._hidden_size = hidden_size
        # self._dropout = dropout
        with self.name_scope():
            self.rnn_cells = nn.HybridSequential()
            self.rnn_cells.add(rnn.BidirectionalCell(
                l_cell= self._cell_type(hidden_size = self._hidden_size,
                i2h_weight_initializer = i2h_weight_initializer,
                h2h_weight_initializer = h2h_weight_initializer,
                i2h_bias_initializer = i2h_bias_initializer,
                h2h_bias_initializer = h2h_bias_initializer,
                prefix = 'rnn_l_'),
                r_cell= self._cell_type(hidden_size = self._hidden_size,
                i2h_weight_initializer = i2h_weight_initializer,
                h2h_weight_initializer = h2h_weight_initializer,
                i2h_bias_initializer = i2h_bias_initializer,
                h2h_bias_initializer = h2h_bias_initializer,
                prefix = 'rnn_r_')))

    def __call__(self, inputs, states = None, valid_length = None):
        return super(SUMEncoder, self).__call__(inputs, states, valid_length)

    def forward(self, inputs, states = None, valid_length = None):
        _, length, _ = inputs.shape
        outputs, new_state = self.rnn_cells[0].unroll(
            length = length, inputs = inputs, begin_state = None, merge_outputs = None,
            valid_length = valid_length, layout = 'NTC')
        if valid_length is not None:
            outputs = mx.nd.SequenceMask(outputs, sequence_length= valid_length, use_sequence_length=True, axis = 1)

        return [outputs, [new_state]]

class Seq2SeqDecoder(Block):
    pass

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
                prefix = 'rnn_'
                )
            )

    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length = None):
        enc_states, rnn_states = encoder_outputs
        batch_size, attn_length, num_hidden = enc_states.shape
        attention_vec = mx.nd.zeros(shape = (batch_size, num_hidden), ctx = enc_states.context)
        decoder_states = [rnn_states, attention_vec, enc_states]
        if encoder_valid_length is not None:
            attn_masks = mx.nd.broadcast_lesser(
                mx.nd.arrange(attn_length, ctx = encoder_valid_length.context).reshape((1,-1)),
                encoder_valid_length.reshape((-1, 1)))
            decoder_states.append(attn_masks)
        return decoder_states

    def decode_seq(self, inputs, states, valid_length = None):
        length = inputs.shape[1]
        output = []
        # additional_outputs = []
        inputs = _as_list(mx.nd.split(inputs, num_outputs = length, axis = 1, squeeze_axis = True))
        rnn_states_1 = []
        # attention_output_1 = []
        context_vec = mx.nd.zeros(length)

        for i in range(length):
            input_size = inputs[i].shape[2]
            x = linear(inputs[i], context_vec, input_size)
            cell_output, states = self.forward(x, states)
            # rnn_states_1.append(states[0])

            output = linear([cell_output] + [context_vec], output_size)
        outputs.append(output)

        return outputs, state, attn_dist

        pass

    def __call__(self, step_input, states):
        return self.forward(step_input, states)


    # def forward(self, step_input, states):
    #     step_output, new_states, step_additional_outputs = super(SUMDecoder,self).forward(step_input,states)

    def attention(self, step_input, enc_states):
        pass

    def linear(self, cell_output, context_vec, ouput_size):
        pass

    def hybrid_forward(self, F, step_input, states):
        context_vec, attention_dist = self.attention_cell(step_input, enc_states)
        outputs, new_states = self.rnn_cells[0].hybrid_forward(step_input, states)
        return outputs, new_states, context_vec, attention_dist



        pass


def get_summ_encoder_decoder(cell_type = 'lstm', num_layers = 1, num_bi_layers = 1, hidden_size = 128, dropout= 0.0,
                            i2h_weight_initializer = i2h_weight_initializer,
                            h2h_weight_initializer = h2h_weight_initializer,
                            i2h_bias_initializer = i2h_bias_initializer,
                            h2h_bias_initializer = h2h_bias_initializer,
                            prefix = 'Summarization_', params = None):
    encoder = SUMEncoder(cell_type = cell_type, num_layers = num_layers, num_bi_layers = num_bi_layers,
                        hidden_size = hidden_size, dropout= dropout,
                        i2h_weight_initializer = i2h_weight_initializer,
                        h2h_weight_initializer = h2h_weight_initializer,
                        i2h_bias_initializer = i2h_bias_initializer,
                        h2h_bias_initializer = h2h_bias_initializer,
                        prefix = prefix + 'enc_', params = params)

    encoder = SUMEncoder(cell_type = cell_type, num_layers = num_layers, num_bi_layers = num_bi_layers,
                        hidden_size = hidden_size, dropout= dropout,
                        i2h_weight_initializer = i2h_weight_initializer,
                        h2h_weight_initializer = h2h_weight_initializer,
                        i2h_bias_initializer = i2h_bias_initializer,
                        h2h_bias_initializer = h2h_bias_initializer,
                        prefix = prefix + 'dec_', params = params)

    return encoder, decoder
