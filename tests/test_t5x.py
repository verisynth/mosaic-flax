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
        res = eval_fn(params, cur_token, past_key_values=cache, use_cache=True)
        # new_state = DecodingState()
        return res[0][:, -1], {}

    # init_cache = [(jnp.zeros((1, 2, 20, 128)), jnp.zeros((1, 2, 20, 128))) for _ in range(32)]
    def preprocess_tokens(tokens, pad_to: int = 10):
        assert pad_to >= tokens.shape[1]
        return jnp.concatenate(
            (jnp.zeros((tokens.shape[0], 1)),
             tokens,
             jnp.zeros((tokens.shape[0], pad_to - tokens.shape[1]))
             ), axis=1).astype(int)

    init_cache = [
        tuple([jnp.zeros((tokens.shape[0], 10, model.n_heads, model.d_model // model.n_heads)) for _ in range(2)])
        for _ in range(model.n_layers)]

    print(preprocess_tokens(tokens, pad_to=10))
    print(temperature_sample(preprocess_tokens(tokens, pad_to=10), init_cache, tokens_to_logits, 100277,
                             initial_index=jnp.ones((tokens.shape[0],), dtype=jnp.int32)))



if __name__ == '__main__':
    test_t5x()
