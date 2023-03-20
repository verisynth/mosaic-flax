from mosaicgpt_flax import FlaxMosaicGPT
from omegaconf import OmegaConf
import tiktoken
import jax
import jax.numpy as jnp


def test_mosaic_gpt():
    cfg = OmegaConf.load("tests/90m.yaml")
    model = FlaxMosaicGPT(cfg=cfg.model)
    assert model is not None

    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = jnp.array(tokenizer.encode_batch(["Hello world!"], allowed_special="all", disallowed_special=()))

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, tokens, training=False)
    print(jax.tree_map(lambda x: x.shape, params))
    assert model.apply(params, tokens)[0].shape == (1, 3, model.vocab_size)
