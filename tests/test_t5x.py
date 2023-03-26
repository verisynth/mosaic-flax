from omegaconf import OmegaConf
from t5x.decoding import temperature_sample, beam_search, DecodingState
import jax
import jax.numpy as jnp
import tiktoken

from mosaicgpt_flax import FlaxMosaicGPT
from mosaicgpt_flax.utils import tokens_to_logits_fn_factory, preprocess_tokens, generate_factory


def test_t5x():
    cfg = OmegaConf.load("tests/90m.yaml")
    model = FlaxMosaicGPT(cfg=cfg.model)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = jnp.array(tokenizer.encode_batch(["Hello world!"], allowed_special="all", disallowed_special=()))

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, tokens, training=False)
    print(jax.tree_map(lambda x: x.shape, params))

    n_heads, d_model, n_layers = cfg.model.n_heads, cfg.model.d_model, cfg.model.n_layers
    init_cache = [
        tuple([jnp.zeros((tokens.shape[0], 10, n_heads, d_model // n_heads)) for _ in range(2)])
        for _ in range(n_layers)]

    print(preprocess_tokens(tokens, pad_to=10))
    generate = generate_factory(model, eos_id=100277, temperature=0.9, topk=128, topp=0.9)
    rng = jax.random.PRNGKey(0)
    print(generate(params, rng, preprocess_tokens(tokens, pad_to=10), init_cache))



if __name__ == '__main__':
    test_t5x()
