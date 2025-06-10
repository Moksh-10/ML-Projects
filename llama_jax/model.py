import jax
import jax.numpy as jnp
from jax import random
import math


def rms_norm(x, wei, eps=1e-5):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * wei * jnp.reciprocal(jnp.sqrt(var + eps))
