"""
The diffusion Scrödinger bridge implementation.

Based on https://github.com/zgbkdlm/fbs/blob/main/fbs/dsb/base.py.
"""
import jax
import jax.numpy as jnp
from gdcs.typings import JArray, JKey, FloatScalar, JFloat
from typing import Callable


def ipf_loss_cont(key: JKey,
                  param: JArray,
                  simulator_param: JArray,
                  init_samples: JArray,
                  ts: JArray,
                  parametric_drift: Callable[[JArray, FloatScalar, JArray], JArray],
                  simulator_drift: Callable[[JArray, FloatScalar, JArray], JArray],
                  dispersion: Callable) -> JFloat:
    r"""The iterative proportional fitting (continuous version) loss used in Schrödinger bridge.
    Proposition 29, de Bortoli et al., 2021.

    Parameters
    ----------
    key : JKey
        A JAX random key.
    param : JArray
        The parameter of the parametric drift function that you wish to learn.
    simulator_param : JArray
        The parameter of the simulation process drift function.
    init_samples : JArray (m, n, ...)
        Samples from the initial distribution (i.e., either the target or the reference depending on if you are
        learning the forward or the backward process).
    ts : JArray (n, )
        Either the forward times `t_0, t_1, ..., t_n` or its reversal, depending on if you are using this function
        to learn the backward or forward process.
    parametric_drift : Callable
        The parametric drift function whose signature is `f(x, t, param)`.
    simulator_drift : Callable
        The simulator process' drift function whose signature is `g(x, t, simulator_param)`.
    dispersion : Callable
        The dispersion function, a function of time.

    Returns
    -------
    JFloat
        The loss.

    Notes
    -----
    When using this function to learn the backward process `target <- ref`,
    simulate the forward process defined by `simulator_drift` at forward `ts`.

    When using this function to learn the forward process `target -> ref`,
    simulate the backward process defined by `simulator_drift` at backward `ts`.
    """
    nsteps = ts.shape[0] - 1
    fn = lambda z, t, dt: z + simulator_drift(z, t, simulator_param) * dt

    def scan_body(carry, elem):
        z, err = carry
        t, t_next, rnd = elem

        dt = jnp.abs(t_next - t)
        z_next = z + simulator_drift(z, t, simulator_param) * dt + jnp.sqrt(dt) * dispersion(t) * rnd
        err = err + jnp.mean(
            (parametric_drift(z_next, t_next, param) * dt - (fn(z, t, dt) - fn(z_next, t, dt))) ** 2)
        return (z_next, err), None

    key, subkey = jax.random.split(key)
    rnds = jax.random.normal(subkey, (nsteps, *init_samples.shape))
    (_, err_final), _ = jax.lax.scan(scan_body, (init_samples, 0.), (ts[:-1], ts[1:], rnds))
    return jnp.mean(err_final / nsteps)
