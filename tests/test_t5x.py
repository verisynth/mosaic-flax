from omegaconf import OmegaConf
from t5x.decoding import temperature_sample, beam_search, DecodingState
import jax
import jax.numpy as jnp
import tiktoken

from mosaicgpt_flax import FlaxMosaicGPT


def test_t5x():
    cfg = OmegaConf.load("tests/90m.yaml")
    model = FlaxMosaicGPT(cfg=cfg.model)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = jnp.array(tokenizer.encode_batch(["Hello world!"], allowed_special="all", disallowed_special=()))

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, tokens, training=False)
    print(jax.tree_map(lambda x: x.shape, params))
    eval_fn = jax.jit(model.apply)

    def tokens_to_logits(decoding_state):
        cur_index = decoding_state.cur_index
        print(decoding_state)
        cur_token = decoding_state.cur_token
        sequences = decoding_state.sequences
        cache = decoding_state.cache
        res = eval_fn(params, sequences, past_key_values=None, use_cache=False)
        # new_state = DecodingState()
        return res[0][:, -1], {}

    # init_cache = [(jnp.zeros((1, 2, 20, 128)), jnp.zeros((1, 2, 20, 128))) for _ in range(32)]
    print(tokens)
    print(temperature_sample(tokens, {}, tokens_to_logits, 100277))


if __name__ == '__main__':
    test_t5x()
