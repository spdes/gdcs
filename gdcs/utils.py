import jax
import jax.numpy as jnp
from gdcs.typings import JArray, JKey
from typing import Callable


def euler_maruyama(key: JKey, x0: JArray, ts: JArray,
                   drift: Callable, dispersion: Callable,
                   integration_nsteps: int = 1,
                   return_path: bool = False) -> JArray:
    r"""Simulate an SDE using the Euler-Maruyama method.

    Parameters
    ----------
    key : JKey
        JAX random key.
    x0 : JArray (..., )
        Initial value.
    ts : JArray (n + 1, )
        Times :math:`t_0, t_1, \ldots, t_n`.
    drift : Callable (..., ), float -> (..., )
        The drift function.
    dispersion : Callable float -> float
        The dispersion function.
    integration_nsteps : int, default=1
        The number of integration steps between each step.
    return_path : bool, default=False
        Whether return the path or just the terminal value.

    Returns
    -------
    JArray (..., ) or JArray (n + 1, ...)
        The terminal value at :math:`t_n`, or the path at :math:`t_0, \ldots, t_n`.
    """
    keys = jax.random.split(key, num=ts.shape[0] - 1)

    def step(xt, t, t_next, key_):
        def scan_body_(carry, elem):
            x = carry
            rnd, t_ = elem
            x = x + drift(x, t_) * ddt + dispersion(t_) * jnp.sqrt(ddt) * rnd
            return x, None

        ddt = jnp.abs(t_next - t) / integration_nsteps
        rnds = jax.random.normal(key_, (integration_nsteps, *x0.shape))
        return jax.lax.scan(scan_body_, xt, (rnds, jnp.linspace(t, t_next - ddt, integration_nsteps)))[0]

    def scan_body(carry, elem):
        x = carry
        key_, t, t_next = elem

        x = step(x, t, t_next, key_)
        return x, x if return_path else None

    terminal_val, path = jax.lax.scan(scan_body, x0, (keys, ts[:-1], ts[1:]))

    if return_path:
        return jnp.concatenate([x0[jnp.newaxis], path], axis=0)
    else:
        return terminal_val
