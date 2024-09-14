"""
Neural networks with FLAX.

Implementations based on https://github.com/zgbkdlm/fbs/blob/main/fbs/dsb/base.py.
"""
import math
import jax
import flax.linen as nn
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from gdcs.typings import JArray, FloatScalar, JKey
from typing import Union, Tuple, Callable, Sequence

nn_param_init = nn.initializers.xavier_normal()


def make_st_nn(key: JKey, neural_network: nn.Module, dim_in: Sequence[int], batch_size: int) -> Tuple[
    JArray, Callable[[JArray], dict], Callable[[JArray, FloatScalar, JArray], JArray]]:
    """Make a neural network for approximating a spatial-temporal function :math:`f(x, t)`.

    Parameters
    ----------
    neural_network : linen.Module
        A neural network instance.
    dim_in : (int, ...)
        The spatial dimension.
    batch_size : int
        The data batch size.
    key : JKey
        A JAX random key.

    Returns
    -------
    JArray, Callable[[JArray], dict], Callable (..., d), (p, ) -> (..., d)
        The initial parameter array, the array-to-dict ravel function, and the NN forward pass evaluation function.
    """
    dict_param = neural_network.init(key, jnp.ones((batch_size, *dim_in)), jnp.ones((batch_size,)))
    array_param, array_to_dict = ravel_pytree(dict_param)

    def forward_pass(x: JArray, t: FloatScalar, param: JArray) -> JArray:
        """The NN forward pass.
        x : (..., d)
        t : (...)
        param : (p, )
        return : (..., d)
        """
        return neural_network.apply(array_to_dict(param), x, t)

    return array_param, array_to_dict, forward_pass


def sinusoidal_embedding(k: Union[JArray, FloatScalar], out_dim: int = 64, max_period: int = 10_000) -> JArray:
    """The so-called sinusoidal positional embedding.

    Parameters
    ----------
    k : FloatScalar
        A time variable. Note that this is in the discrete time.
    out_dim : int
        The output dimension.
    max_period : int
        The maximum period.

    Returns
    -------
    JArray (..., out_dim)
        An array.
    """
    half = out_dim // 2

    fs = jnp.exp(-math.log(max_period) * jnp.arange(half) / (half - 1))
    embs = k * fs
    embs = jnp.concatenate([jnp.sin(embs), jnp.cos(embs)], axis=-1)
    if out_dim % 2 == 1:
        raise NotImplementedError(f'out_dim is implemented for even number only, while {out_dim} is given.')
    return embs


class _CrescentTimeBlock(nn.Module):
    dt: float
    nfeatures: int

    @nn.compact
    def __call__(self, time_emb):
        time_emb = nn.Dense(features=self.nfeatures, kernel_init=nn_param_init)(time_emb)
        time_emb = nn.gelu(time_emb)
        time_emb = nn.Dense(features=self.nfeatures, kernel_init=nn_param_init)(time_emb)
        return time_emb


class CrescentMLP(nn.Module):
    """The MLP neural network construction used in the pedagogical example.
    """
    dt: float
    dim_out: int = 3
    hiddens = [64, 32, 16]

    @nn.compact
    def __call__(self, x, t):
        k = t / self.dt
        if t.ndim < 1:
            time_emb = jnp.expand_dims(sinusoidal_embedding(k, out_dim=64), 0)
        else:
            time_emb = jax.vmap(lambda z: sinusoidal_embedding(z, out_dim=64))(k)

        for h in self.hiddens:
            x = nn.Dense(features=h, kernel_init=nn_param_init)(x)
            x = (x * _CrescentTimeBlock(dt=self.dt, nfeatures=h)(time_emb) +
                 _CrescentTimeBlock(dt=self.dt, nfeatures=h)(time_emb))
            x = nn.gelu(x)
        x = nn.Dense(features=self.dim_out, kernel_init=nn_param_init)(x)
        return jnp.squeeze(x)
