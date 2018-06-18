class BeamSearchSummarizer(object):
    def __init__(self, model, beam_size = 4, scorer = BeamSearchScorer(), max_length = 100):
        self._model = model
        self._sampler = BeamSearchSampler(decoder = self._decode_logprob, beam_size = beam_size,
                                        eos_id = model.vocab.token_to_idx[model.vocab.eos_token],
                                        scorer = scorer,
                                        max_length = max_length)


    def _decode_logprob(self, self_input, states):
        #logprob: (batch_size, V)
        out, states = self._model.decode_step(step_input, states)
        return mx.nd.log_softmax(out), states
