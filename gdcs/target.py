"""
The test problem in the pedagogical example.
"""
import jax
import jax.numpy as jnp
from gdcs.typings import JArray, JKey, FloatScalar
from typing import Tuple


class Crescent:
    r"""A crescent-shaped posterior distribution.

    X ~ GM(m0, v0, w0, m1, v1, w1)
    Y | X ~ N(Y | X_1 / c + 0.5 * (X_0 ^ 2 + c ^ 2), xi)
    """

    def __init__(self, c: float, xi: float):
        self.c = c
        self.xi = xi

        self.w0 = 0.5
        self.m0 = jnp.array([0., 0.])
        self.v0 = jnp.array([[1., 0.8],
                             [0.8, 1.]])
        self.chol_v0 = jnp.linalg.cholesky(self.v0)

        self.w1 = 1 - self.w0
        self.m1 = -self.m0
        self.v1 = jnp.array([[1., -0.8],
                             [-0.8, 1.]])
        self.chol_v1 = jnp.linalg.cholesky(self.v1)

        self.ws = jnp.array([self.w0, self.w1])
        self.ms = jnp.concatenate([self.m0[None, :], self.m1[None, ::]], axis=0)
        self.vs = jnp.concatenate([self.v0[None, ...], self.v1[None, ...]], axis=0)
        self.chols = jnp.concatenate([self.chol_v0[None, ...], self.chol_v1[None, ...]], axis=0)

    def sampler_x(self, key: JKey):
        key_cat, key_x = jax.random.split(key)
        ind = jax.random.choice(key_cat, 2, p=self.ws)
        return self.ms[ind] + self.chols[ind] @ jax.random.normal(key_x, shape=(2,))

    def emission(self, x):
        return x[..., 1] / self.c + 0.5 * (x[..., 0] ** 2 + self.c ** 2)

    def sampler_y_cond_x(self, key: JKey, x: JArray):
        m = self.emission(x)
        return m + jnp.sqrt(self.xi) * jax.random.normal(key, shape=m.shape)

    def sampler_joint(self, key: JKey) -> Tuple[JArray, JArray]:
        key_x, key_y = jax.random.split(key)
        x = self.sampler_x(key_x)
        y = self.sampler_y_cond_x(key_y, x)
        return x, y

    def logpdf_x(self, x):
        log_pdf0 = jax.scipy.stats.multivariate_normal.logpdf(x, self.ms[0], self.vs[0])
        log_pdf1 = jax.scipy.stats.multivariate_normal.logpdf(x, self.ms[1], self.vs[1])
        return jax.scipy.special.logsumexp(jnp.array([jnp.log(self.w0) + log_pdf0,
                                                      jnp.log(self.w1) + log_pdf1]))

    def logpdf_y_cond_x(self, y, x):
        return jax.scipy.stats.norm.logpdf(y, self.emission(x), jnp.sqrt(self.xi))

    def pdf_x_cond_y(self, xs_mesh: JArray, y: FloatScalar) -> JArray:
        """

        Parameters
        ----------
        xs_mesh : JArray (n1, n2, d)
        y : FloatScalar

        Returns
        -------
        JArray (n1, n2)
            The PDF evaluated at the Cartesian grids.
        """

        def unnormalised_joint(x_):
            return jnp.exp(self.logpdf_y_cond_x(y, x_) + self.logpdf_x(x_))

        evals = jax.vmap(jax.vmap(unnormalised_joint, in_axes=[0]), in_axes=[0])(xs_mesh)
        ell = jax.scipy.integrate.trapezoid(jax.scipy.integrate.trapezoid(evals, xs_mesh[0, :, 0], axis=0),
                                            xs_mesh[:, 0, 1])
        return evals / ell
