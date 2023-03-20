from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from omegaconf import DictConfig


from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

# TODO:
#  1. attn_mask,
#  2. different attn_impl


class FlaxAttention(nn.Module):
    """
    Flax implementation of attention.
    """
    dtype: jnp.dtype = jnp.float32
    causal: bool = True  # can't think of a case where we want non-causal attention...
    cfg: DictConfig = None

    def setup(self):
        if self.cfg is None:
            raise ValueError("Must provide a config for attention.")

        self.d_model = self.cfg.d_model
        self.n_heads = self.cfg.n_heads
        self.n_layers = self.cfg.n_layers
        assert self.d_model % self.n_heads == 0, \
            f"Number of hidden dim must be divisible by number of heads. Got {self.d_model=} and {self.n_heads=}."
        self.attn_qk_ln = self.cfg.get('attn_qk_ln', True)
        self.attn_clip_qkv = self.cfg.get('attn_clip_qkv', False)
        self.no_bias = self.cfg.get('no_bias', True)

        self.W_qkv = nn.Dense(3 * self.d_model,
                              use_bias=not self.no_bias)
        if self.attn_qk_ln:
            self.q_ln = nn.LayerNorm(self.d_model, use_bias=not self.no_bias)
            self.k_ln = nn.LayerNorm(self.d_model, use_bias=not self.no_bias)

        self.out_proj = nn.Dense(self.d_model,
                                 use_bias=not self.no_bias)

    def _get_causal_mask(self, qs, ks):
        assert ks >= qs, "key_len must be greater than or equal to query_len"
        return jnp.triu(jnp.ones((qs, ks), dtype=bool), ks - qs + 1)

    def _attn_fn(self, query, key, value) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Eager implementation of attention. c.f. `attention_ref(...)` in
        `https://github.com/HazyResearch/flash-attention/blob/main/tests/test_flash_attn.py`.
        It is still off by a minor amount when comparing with the flash attention, but should not be practically
        a problem.

        query: [..., query_len, num_heads, head_dim]
        key: [..., key_len, num_heads, head_dim]
        value: [..., value_len, num_heads, head_dim]
        (if pmap/pjit is applied, ... could have 2 dims involving data-parallel in addition to batch dim)

        key_len is normally the same as value_len, but could possibly larger than query_len (implementing kv cache).
        """
        attn_weights = jnp.einsum('...shd,...thd->...hst', query / jnp.sqrt(query.shape[-1]), key)

        if self.causal:
            causal_mask = self._get_causal_mask(query.shape[-3], key.shape[-3])
            attn_weights = jnp.where(causal_mask[None, None, :, :], attn_weights, float('-inf'))

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum('...hst,...thd->...shd', attn_weights, value)
        return attn_output, attn_weights

    def __call__(self, x, attn_mask=None, layer_past=None, use_cache=False, training=False)\
            -> tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        assert x.shape[-1] == self.d_model, \
            f"Input to Attention layer has different dimension than the hidden dimension. Got {x.shape[-1]}"
        qkv = self.W_qkv(x)
        if self.attn_clip_qkv:
            qkv = jnp.clip(qkv, a_min=-self.clip_qkv, a_max=self.clip_qkv)

        q, k, v = jnp.split(qkv, 3, axis=-1)

        if self.attn_qk_ln:
            # Applying layernorm to qk
            dtype = qkv.dtype
            q = self.q_ln(q).astype(dtype)
            k = self.k_ln(k).astype(dtype)

        n_heads = self.n_heads
        head_dim = self.d_model // self.n_heads
        q, k, v = q.reshape(q.shape[:-1] + (n_heads, head_dim)), \
                  k.reshape(k.shape[:-1] + (n_heads, head_dim)), \
                  v.reshape(v.shape[:-1] + (n_heads, head_dim))

        if layer_past is not None:
            assert len(layer_past) == 2, 'layer_past should be a tuple of (k, v)'
            past_key, past_value = layer_past
            k = jnp.concatenate((past_key, k), dim=1)
            v = jnp.concatenate((past_value, v), dim=1)

        if use_cache is True:
            present = (k, v)
        else:
            present = None

        context, attn_weights = self._attn_fn(q, k, v)

        return self.out_proj(context.reshape(context.shape[:-2] + (self.d_model,))), attn_weights, present


class FlaxGPTMLP(nn.Module):
    cfg: DictConfig = None

    def setup(self):
        if self.cfg is None:
            raise ValueError("Must provide a config for attention.")

        self.d_model = self.cfg.d_model
        self.no_bias = self.cfg.get('no_bias', True)
        self.mlp_ratio = self.cfg.get('mlp_ratio', 1.0)

        self.mlp_up = nn.Dense(int(self.mlp_ratio * self.d_model),
                               use_bias=not self.no_bias)
        self.mlp_act = nn.gelu
        self.mlp_down = nn.Dense(self.d_model,
                                 use_bias=not self.no_bias)

    def __call__(self, x, training=False):
        assert x.shape[-1] == self.d_model, \
            f"Input to MLP layers have different dimensions than the hidden dimension. Got {x.shape[-1]}"
        return self.mlp_down(self.mlp_act(self.mlp_up(x)))


class FlaxGPTBlock(nn.Module):
    cfg: DictConfig = None

    def setup(self) -> None:
        if self.cfg is None:
            raise ValueError("Must provide a config for attention.")

        self.d_model = self.cfg.d_model
        self.resid_pdrop = self.cfg.get('resid_pdrop', 0.0)
        self.no_bias = self.cfg.get('no_bias', True)

        self.ln_1 = nn.LayerNorm(self.d_model, use_bias=not self.no_bias)
        self.causal_attn = FlaxAttention(cfg=self.cfg)
        self.ln_2 = nn.LayerNorm(self.d_model, use_bias=not self.no_bias)
        self.mlp = FlaxGPTMLP(cfg=self.cfg)
        self.resid_attn_dropout = nn.Dropout(self.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(self.resid_pdrop)

    def __call__(self, x, attn_mask=None, layer_past=None, use_cache=False, training=False) -> tuple:
        a = self.ln_1(x)
        b, _, present = self.causal_attn(a, attn_mask,
                                         layer_past=layer_past, use_cache=use_cache, training=training)

        x = x + self.resid_attn_dropout(b, deterministic=not training)
        m = self.ln_2(x)
        n = self.mlp(m, training=training)
        x = x + self.resid_mlp_dropout(n, deterministic=not training)
        return x, present


class FlaxMosaicGPT(nn.Module):
    cfg: DictConfig = None
    vocab_size: int = 10027

    def setup(self) -> None:
        assert self.cfg.name == 'mosaic_gpt', f'Tried to build MosaicGPT model with cfg.name={self.cfg.name}'

        self.max_seq_len = self.cfg.max_seq_len
        self.emb_pdrop = self.cfg.get('emb_pdrop', 0.0)
        self.d_model = self.cfg.d_model
        self.n_heads = self.cfg.n_heads
        self.n_layers = self.cfg.n_layers
        self.attn_qk_ln = self.cfg.get('attn_qk_ln', True)
        self.attn_clip_qkv = self.cfg.get('attn_clip_qkv', False)
        self.weight_tied = self.cfg.get('weight_tied', True)

        # todo: slightly skeptical about whether this is necessary. Leaving it alone for now
        self.embedding_fraction = self.cfg.get('embedding_fraction', 1)

        self.no_bias = self.cfg.get('no_bias', True)

        self.wte = nn.Embed(self.vocab_size,
                            self.d_model)
        self.wpe = nn.Embed(self.max_seq_len,
                            self.d_model)
        self.emb_drop = nn.Dropout(self.emb_pdrop)

        self.blocks = [FlaxGPTBlock(self.cfg) for _ in range(self.n_layers)]

        self.ln_f = nn.LayerNorm(self.d_model, use_bias=not self.no_bias)

        if not self.weight_tied:
            self.out = nn.Dense(self.vocab_size, use_bias=not self.no_bias)

    def __call__(self,
                input_ids: jnp.ndarray,
                past_key_values: Optional[jnp.ndarray] = None,
                use_cache: bool = False,
                training: bool = False) -> tuple[jnp.ndarray, Optional[list[jnp.ndarray]]]:
        """
        Note on kv cache: if use_cache is True, it will return a tuple: logits, present_key_values
            otherwise, it will return logits only.
        """
        assert len(input_ids.shape) in [2, 3], f"input_ids dimension must be 2. Got {len(input_ids.shape)}."
        batch_size, current_seq_len = input_ids.shape

        if past_key_values is None:
            past_key_values = [None] * self.n_layers
            past_position = 0
        else:
            assert len(past_key_values) == self.n_layers, \
                f"length of past_key_values must match the number of layers. Got {len(past_key_values)}."
            # get the key tensor whose spec should be (batch, seq, n_head, head_dim), and
            # collect the `seq`, so that we shift the position embedding later.
            past_position = past_key_values[0][0].shape[1]

        tok_emb = self.wte(input_ids)

        if current_seq_len + past_position > self.max_seq_len:
            raise ValueError(
                f'Cannot forward input with past sequence length {past_position} and current sequence length '
                f'{current_seq_len + 1}, this model only supports total sequence length <= {self.max_seq_len}.'
            )
        pos = jnp.arange(past_position, current_seq_len + past_position, dtype=jnp.int64)[None, :]
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        x = self.emb_drop(x, deterministic=not training)  # type: ignore

        # TODO: Adding attention mask. Should be trivial but I'm lazy.
        attn_mask = None

        present_key_values = []
        for block, layer_past in zip(self.blocks, past_key_values):  # type: ignore
            x, present = block(x, attn_mask, layer_past=layer_past, use_cache=use_cache, training=training)
            present_key_values.append(present)

        x = self.ln_f(x)  # type: ignore
        # output embedding weight tied to input embedding

        if self.weight_tied:
            logits = lax.dot_general(x, self.wte.embedding, (((len(x.shape) - 1,), (1,)), ((), ())))
        else:
            logits = self.out(x)
        if use_cache:
            return logits, present_key_values
        else:
            return logits, None
