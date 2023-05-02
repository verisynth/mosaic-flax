"""
This is a gradio app to perform inferences on a tpu pod
"""

from mosaicgpt_flax import FlaxMosaicGPT
from mosaicgpt_flax import read_torch_checkpoint
from mosaicgpt_flax.utils import preprocess_tokens, generate_factory, populate_inputs
import tiktoken
from omegaconf import OmegaConf
import numpy as np
import jax
import jax.numpy as jnp
import absl
from absl import app, flags
from jax.experimental.pjit import pjit, with_sharding_constraint
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils
import gradio as gr

# Start a device mesh.
device_mesh = mesh_utils.create_device_mesh((2, 4))
print(device_mesh)
# Annotate each axis with a name.
mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))


FLAGS = flags.FLAGS
flags.DEFINE_string('cfg_path', 'tests/3b.yaml', 'Path to config file', short_name='c')
flags.DEFINE_string('model_path', 'ckpt.pt', 'Path to checkpoint file', short_name='m')


def main(argv):

    cfg = OmegaConf.load(FLAGS.cfg_path)

    params = read_torch_checkpoint(FLAGS.model_path, dtype=jnp.bfloat16)
    params = jax.tree_util.tree_map(lambda x: jax.device_put_replicated(x, jax.devices()), params)

    model = FlaxMosaicGPT(cfg=cfg.model)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    n_heads, d_model, n_layers = cfg.model.n_heads, cfg.model.d_model, cfg.model.n_layers

    compiled = {}  # cache the jit-compiled functions

    def inference(prefix: str = "", max_len: int = 16, temp=0.35, top_k=64, top_p=0.95, rng=0):
        tokens = jnp.array(tokenizer.encode_batch([prefix], allowed_special="all", disallowed_special=()))
        max_len = max(int(max_len + tokens.shape[1]), 1 + tokens.shape[1])

        init_cache = [
            tuple([jnp.zeros((tokens.shape[0], max_len, n_heads, d_model // n_heads)) for _ in range(2)])
            for _ in range(n_layers)]
        tokens = preprocess_tokens(tokens, max_len)

        if (temp, top_k, top_p) not in compiled:
            compiled[(temp, top_k, top_p)] = jax.pmap(generate_factory(model,
                                                                       eos_id=100277, temperature=temp,
                                                                       topk=top_k, topp=top_p))
        generate = compiled[(temp, top_k, top_p)]

        if rng == 0:
            rng = np.random.randint(0, 2 ** 32 - 1, dtype=np.uint32)
        rng = jax.random.PRNGKey(rng)
        rngs, tokens, init_cache = populate_inputs(rng, tokens, init_cache)

        res = generate(params, rngs, tokens, init_cache)

        return [tokenizer.decode(list(res[0][i, 0, 0, :])) for i in range(8)]

    demo = gr.Interface(
        inference,
        [
            "textbox",
            "number",
            "number",
            "number",
            "number",
            "number"
        ],
        ["text"],
        title="Generate",
        description="Gen",
    )
    demo.launch(share=True)


if __name__ == '__main__':
    app.run(main)
