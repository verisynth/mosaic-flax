import flax
import torch
from t5x.decoding import temperature_sample

from mosaicgpt_flax import FlaxMosaicGPT

import jax
import jax.numpy as jnp


def add_item(d, keys, value):
    if not keys:
        return
    if len(keys) == 1:
        d[keys[0]] = value
    else:
        if keys[0] not in d:
            d[keys[0]] = {}
        add_item(d[keys[0]], keys[1:], value)


def map_name(key_1, key_2):
    if key_2 == 'bias':
        return 'bias'
    elif key_1.endswith('ln') or key_1.startswith('ln'):
        return 'scale'  # weight->scale for layer norms
    else:
        return 'kernel'  # weight->kernel for dense layers


def convert_data(state: torch.tensor, embedding=False, dtype=jnp.float32) -> jnp.ndarray:
    if len(state.shape) == 1 or embedding:
        return jnp.array(state.numpy(), dtype=dtype)
    elif len(state.shape) == 2 and not embedding:
        return jnp.array(state.numpy(), dtype=dtype).T
    else:
        raise ValueError("unrecognized shape: ", state.shape)


def map_layer_name(key):
    return key.replace('W_qkv', 'Wqkv')


def convert_weights(key, splits, new_state_dict, state_dict, dtype=jnp.float32):
    if splits[0].startswith('blocks'):
        if splits[2].endswith('attn') or splits[2] == 'mlp':
            # attn, causal_attn, and mlp have other layers inside like attn.W_qkv.weight
            layer_name = 'mlp' if splits[2] == 'mlp' else 'causal_attn'
            name = map_name(splits[3], splits[4])
            add_item(new_state_dict, [splits[0] + '_' + splits[1], layer_name, map_layer_name(splits[3]), name],
                     convert_data(state_dict[key], dtype=dtype))
        else:
            # others
            name = map_name(splits[2], splits[3])
            add_item(new_state_dict, [splits[0] + '_' + splits[1], splits[2], name],
                     convert_data(state_dict[key], dtype=dtype))
    elif splits[0] in ['wte', 'wpe']:
        add_item(new_state_dict, [splits[0], 'embedding'], convert_data(state_dict[key], embedding=True))
    else:
        name = map_name(splits[0], splits[1])
        add_item(new_state_dict, [map_layer_name(splits[0]), name], convert_data(state_dict[key], dtype=dtype))


def read_torch_checkpoint(path: str, dtype=jnp.float32) -> flax.core.FrozenDict:
    """
    Read pytorch checkpoint and convert it to flax FrozenDict.
    The checkpoint file is loaded by torch.load and the state dict is nested under 'state'->'model' keys.
    :param path: checkpoint path
    :param dtype: the dtype of the parameters
    :return: a FrozenDict containing parameters
    """
    state_dict = torch.load(path, map_location='cpu')['state']['model']
    new_state_dict = {}
    offset = None
    for key in state_dict:
        splits = key.split('.')
        if offset is None:
            if splits[0] == 'transformer':
                offset = 1
            elif splits[0] == 'model' and splits[1] == 'transformer':
                offset = 2
            else:
                offset = 0
        print(f"Converting the layer: {key} to {splits[offset:]}")
        convert_weights(key, splits[offset:], new_state_dict, state_dict, dtype=dtype)
    params = flax.core.FrozenDict({'params': new_state_dict})
    return params


def tokens_to_logits_fn_factory(params, model):
    """
    The factory function for `tokens_to_logits` function. This fits into the T5X library decoding spec.
    :param params: the pytree object of model parameters
    :param model: the flax model (containing a jitt-able `apply` method)
    :return: the `tokens_to_logits` function
    """
    eval_fn = model.apply

    def tokens_to_logits(decoding_state):
        cur_index = decoding_state.cur_index
        cur_token = decoding_state.cur_token
        cache = decoding_state.cache
        res = eval_fn(params, cur_token, past_key_values=cache, past_position=cur_index[0], use_cache=True)
        return res[0][..., -1, :], [(x[0][..., 1:, :, :], x[1][..., 1:, :, :]) for x in res[1]]

    return tokens_to_logits


def preprocess_tokens(tokens, pad_to: int = 10):
    """
    Preprocess the tokens for the generation task.
    Given input tokens, it will be padded by 0 with length 1 in the beginning, and padded by 0
      with length `pad_to - tokens.shape[1]` in the end. Overall, the length of the output tokens
      will be `pad_to + 1`.
    This prepares the tokens for generation tasks to generate sequences of length `pad_to`.
    :param tokens: the input tokens
    :param pad_to: the number of padding to be added in the end
    :return: the padded tokens
      For example,
        tokens = [[1, 2, 3], [4, 5, 6]], pad_to = 5
        return = [[0, 1, 2, 3, 0, 0], [0, 4, 5, 6, 0, 0]]
    """
    assert pad_to >= tokens.shape[1]
    return jnp.concatenate((jnp.zeros((tokens.shape[0], 1)),
                            tokens,
                            jnp.zeros((tokens.shape[0], pad_to - tokens.shape[1]))), axis=1).astype(int)


def generate_factory(model, eos_id=100277, temperature=0.9, topk=128, topp=0.9):
    """
    The factory function for `generate` function. This wraps around the T5X `temperature_sample` function to provide
    a jit-compilable version of generation function.
    :param model: the flax model (containing a jitt-able `apply` method)
    :param eos_id: the id of the end-of-sequence token
    :param temperature: the temperature
    :param topk: top k
    :param topp: top p
    :return: a jit-compilable generation function
      args: params, tokens, cache
        params: the pytree object of model parameters
        rng: the rng key(s)
        tokens: the 0-padded input tokens (e.g., [[0, 1, 2, 3, 0, 0]], 0's at the end will be filled in for generation)
          One may call `preprocess_tokens` to prepare the tokens for generation.
        cache: the 0-padded kv cache for generation. The signature is list[tuple[jnp.ndarray, jnp.ndarray]].
          length of list = num_layers, shape of each jnp.ndarray = (batch_size, max_seq_len, num_head, head_dim)
    """
    if topk != 0:  # prefer top-k sampling
        topp = 0

    def generate(params, rng, tokens, cache):
        tokens_to_logits = tokens_to_logits_fn_factory(params, model)

        return temperature_sample(tokens, cache, eos_id=eos_id,
                                  tokens_to_logits=tokens_to_logits,
                                  decode_rng=rng,
                                  temperature=temperature,
                                  topk=topk,
                                  topp=topp)
    return generate
