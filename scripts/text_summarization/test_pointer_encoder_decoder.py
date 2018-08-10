import mxnet as mx
from encoder_decoder_new import Pointer_Generator_Encoder, Pointer_Generator_Decoder

def test_encoder():
    encoder = Pointer_Generator_Encoder(hidden_size=14)
    print(encoder)
    encoder.initialize(init='One')
    inputs = mx.nd.random.uniform(shape=(5,6,7))
    hidden = encoder.begin_state(func=mx.nd.zeros, batch_size=5)
    output, states = encoder(inputs, hidden)
    assert output.shape == (5,6,28), output.shape
    assert states[0].shape == (2, 5, 14), states[0].shape
    assert states[1].shape == (2, 5, 14), states[0].shape

def test_decoder():
    decoder = Pointer_Generator_Decoder(hidden_size=14)
    print(decoder)
    decoder.initialize(init='One')
    enc_outputs = mx.nd.random.uniform(shape=(5,6, 28))
    h_state = mx.nd.random.uniform(shape=(2,5,14))
    c_state = mx.nd.random.uniform(shape=(2,5,14))
    decoder_input = decoder.init_state_from_encoder([enc_outputs, [h_state, c_state]])
    rnn_states = decoder_input[0]
    assert isinstance(rnn_states, list)
    assert rnn_states[0].shape == (5, 14), rnn_states[0].shape
    assert rnn_states[1].shape == (5, 14), rnn_states[1].shape

    inputs = mx.nd.random.uniform(shape=(5,6,7))
    valid_length = mx.nd.ones(shape=(5,))*6
    outputs_vs, context_vecs, attention_dists = decoder.decode_seq(inputs=inputs, states=decoder_input, valid_length=valid_length)
    assert isinstance(outputs_vs, list)
    assert len(outputs_vs) == 6
    assert outputs_vs[0].shape == (5, 100), outputs_vs[0].shape
    assert isinstance(context_vecs, list)
    assert len(context_vecs) == 6
    assert context_vecs[0].shape == (5, 28), context_vecs[0].shape
    assert isinstance(attention_dists, list)
    assert len(attention_dists) == 6
    assert attention_dists[0].shape == (5, 6), attention_dists[0].shape



if __name__ == '__main__':
    test_encoder()
    test_decoder()
