"""
Generic Feynman--Kac models.
"""
import jax
import jax.numpy as jnp
from gdcs.typings import JArray, JKey, FloatScalar
from typing import Callable, Tuple


def smc_feynman_kac(key: JKey,
                    m0: Callable[[JArray, int], JArray],
                    log_g0: Callable[[JArray, JArray], JArray],
                    mk: Callable[[JArray, JArray, JArray, JArray], JArray],
                    log_gk: Callable[[JArray, JArray, JArray, JArray, JArray, JKey], JArray],
                    y: JArray,
                    ts: JArray,
                    resampling: Callable[[JKey, JArray], JArray],
                    nparticles: int) -> Tuple[JArray, JArray, JArray]:
    """Sequential Monte Carlo simulation of Feynman--Kac models. See Algorithm 4.1 in our paper.

    Parameters
    ----------
    key : JKey
        A JAX random key.
    m0 : Callable
        The initial sampler.
    log_g0 : Callable
        The initial (log) potential function.
    mk : Callable
        The Markov transition kernel at k.
    log_gk : Callable
        The (log) potential function
    y : JArray
        The condition.
    ts : JArray (n, )
        The times (always in the forward direction).
    resampling : Callable
        The resampling scheme.
    nparticles : int
        The number of particles.

    Returns
    -------

    """
    key_init, key_body = jax.random.split(key)

    us0 = m0(key_init, nparticles)
    log_ws_ = log_g0(us0, y)
    log_ws0 = log_ws_ - jax.scipy.special.logsumexp(log_ws_)

    def scan_body(carry, elem):
        us, log_ws = carry
        key_k, t_k, t_km1 = elem
        key_resample, key_markov, key_other = jax.random.split(key_k, num=3)

        inds = resampling(key_resample, jnp.exp(log_ws))
        us = us[inds]

        us_prop = mk(key_markov, us, y, t_km1)
        log_ws_ = log_gk(us_prop, us, y, t_k, t_km1, key_other)
        log_ws = log_ws_ - jax.scipy.special.logsumexp(log_ws_)

        ess = jnp.exp(-jax.scipy.special.logsumexp(log_ws * 2))
        return (us_prop, log_ws), ess

    keys = jax.random.split(key_body, num=ts.shape[0] - 1)
    (usT, log_wsT), esss = jax.lax.scan(scan_body, (us0, log_ws0), (keys, ts[1:], ts[:-1]))
    return usT, log_wsT, esss
