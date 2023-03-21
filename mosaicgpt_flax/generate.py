import functools

from .mosaic_gpt import FlaxMosaicGPT
from .utils import read_torch_checkpoint
import jax
import jax.numpy as jnp
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")


def top_k_filtering(logits, top_k=32, filter_value=-float('Inf')):
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_keep = jax.lax.top_k(logits, top_k)[1]
    logits = jnp.where(jnp.isin(jnp.arange(logits.shape[-1]), indices_to_keep), logits, filter_value)
    return logits


def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    sorted_indices = jnp.argsort(-logits)
    sorted_logits = logits[sorted_indices]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs[:-1] > top_p

    indices_to_remove = sorted_indices[sorted_indices_to_remove + 1]
    logits = logits.at[indices_to_remove].set(filter_value)
    return logits


@jax.jit
def top_k_top_p_filtering(logits: jnp.ndarray,
                          top_k: int = 0,
                          top_p: float = 0.0,
                          filter_value: float = -float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert len(logits.shape) == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = jnp.minimum(top_k, logits.shape[-1])  # Safety check

    logits = jax.lax.cond(top_k > 0,
                          lambda x: top_k_filtering(x, top_k=top_k, filter_value=filter_value),
                          lambda x: x,
                          logits)
    logits = jax.lax.cond(top_p > 0.0,
                          lambda x: top_p_filtering(x, top_p=top_p, filter_value=filter_value),
                          lambda x: x,
                          logits)

    return logits


def generate(params, eval_fn, prompt: str, max_len: int = 100, top_k: int = 0, top_p: float = 0.0, temp: float = 1.0):
    """
    Args:
        params: FrozenDict containing the model parameters
        eval_fn: the evaluation function (usually the `model.apply` or `jax.jit(model.apply)`)
        prompt: the prompt string
        max_len: the max generation length
        top_k: top k
        top_p: top p
        temp: temperature
        the completed string (containing the prompt)
    """
    tokens = tokenizer.encode_batch([prompt], allowed_special="all", disallowed_special=())
    current_state = tokens
    past_key_values = None
    rng = jax.random.PRNGKey(0)
    for _ in range(max_len):
        key, subkey = jax.random.split(rng)
        if past_key_values is None:
            tok = current_state
        else:
            tok = [current_state[0][-1:]]
        outputs, past_key_values = eval_fn(params, jnp.array(tok),
                                           past_key_values=past_key_values, use_cache=True)

        logits = outputs[-1][-1] * 1. / temp
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        out_tk = jax.random.categorical(subkey, logits)

        current_state[0].append(int(out_tk))
    return tokenizer.decode(current_state[0])
