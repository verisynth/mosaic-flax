import flax
import torch
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


def convert_data(state: torch.tensor, embedding=False) -> jnp.ndarray:
    if len(state.shape) == 1 or embedding:
        return jnp.array(state.numpy())
    elif len(state.shape) == 2 and not embedding:
        return jnp.array(state.numpy()).T
    else:
        raise ValueError("unrecognized shape: ", state.shape)


def map_layer_name(key):
    return key.replace('W_qkv', 'Wqkv')


def convert_weights(key, splits, new_state_dict, state_dict):
    if splits[0].startswith('blocks'):
        if splits[2].endswith('attn') or splits[2] == 'mlp':
            # attn, causal_attn, and mlp have other layers inside like attn.W_qkv.weight
            layer_name = 'mlp' if splits[2] == 'mlp' else 'causal_attn'
            name = map_name(splits[3], splits[4])
            add_item(new_state_dict, [splits[0] + '_' + splits[1], layer_name, map_layer_name(splits[3]), name],
                     convert_data(state_dict[key]))
        else:
            # others
            name = map_name(splits[2], splits[3])
            add_item(new_state_dict, [splits[0] + '_' + splits[1], splits[2], name],
                     convert_data(state_dict[key]))
    elif splits[0] in ['wte', 'wpe']:
        add_item(new_state_dict, [splits[0], 'embedding'], convert_data(state_dict[key], embedding=True))
    else:
        name = map_name(splits[0], splits[1])
        add_item(new_state_dict, [map_layer_name(splits[0]), name], convert_data(state_dict[key]))


def read_torch_checkpoint(path: str) -> flax.core.FrozenDict:
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
        convert_weights(key, splits[offset:], new_state_dict, state_dict)
    params = flax.core.FrozenDict({'params': new_state_dict})
    return params
