"""
The Feynman--Kac method for conditional sampling. See Equation (4.4).
"""
import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
from gdcs.target import Crescent
from gdcs.nns import make_st_nn, CrescentMLP
from gdcs.feynman_kac import smc_feynman_kac
from gdcs.resampling import stratified

jax.config.update("jax_enable_x64", False)
key = jax.random.PRNGKey(666)

# Define the data
crescent = Crescent(c=1., xi=0.5)


# The likelihood pi(y | x)
def logpdf_likelihood(y_, x):
    return crescent.logpdf_y_cond_x(y_, x)


# Load the DSB model for pi_X
# Define the parametric neural network
nn_dt = 1. / 200
key, subkey = jax.random.split(key)
my_nn = CrescentMLP(dt=nn_dt, dim_out=2)
_, _, nn_drift = make_st_nn(subkey, neural_network=my_nn, dim_in=(2,), batch_size=2)
param_bwd = np.load('checkpoints/dsb-x-15.npz')['param_bwd']

# Times
T = 1.
nsteps = 128
dt = T / nsteps
ts = jnp.linspace(0., T, nsteps + 1)


def ref_sampler(key_, n: int = 1):
    """The reference distribution is a standard Normal.
    """
    return jax.random.normal(key_, shape=(n, 2))


def rev_drift(u, t):
    """The reversal part of the prior diffusion.
    """
    return nn_drift(u, T - t, param_bwd)


def rev_dispersion(_):
    return 1.


def rev_transition_sampler(key_, us, t_k):
    """The Euler--Maruyama transition of the reversal
    """
    cond_m, cond_scale = us + rev_drift(us, t_k) * dt, math.sqrt(dt) * rev_dispersion(t_k)
    return cond_m + cond_scale * jax.random.normal(key_, shape=us.shape)


# Define the Feynman--Kac model and the SMC sampler. See Equation (4.4).
def m0(key_, nparticles_):
    return ref_sampler(key_, n=nparticles_)


def log_g0(us, y_):
    return log_lk(y_, us, 0., _)


def lam_k(t_k):
    return 1.


def log_lk(y_, us, t_k, _):
    return jax.vmap(logpdf_likelihood, in_axes=[None, 0])(lam_k(t_k) * y_, us)


def mk(key_, us, y_, t_km1):
    return rev_transition_sampler(key_, us, t_km1)


def log_gk(us_k, us_km1, y_, t_k, t_km1, key_):
    key_k, key_km1 = jax.random.split(key_)
    return log_lk(y_, us_k, t_k, key_k) - log_lk(y_, us_km1, t_km1, key_km1)


# Do conditional sampling
y = 5
nparticles = 10000

# samples usT, weights log_wsT, and effective sample sizes esss
key, subkey = jax.random.split(key)
usT, log_wsT, esss = smc_feynman_kac(subkey, m0, log_g0, mk, log_gk, y, ts, stratified, nparticles)

key, subkey = jax.random.split(key)
cond_samples = usT[stratified(subkey, jnp.exp(log_wsT))]
np.save(f'cond_samples_fk_{y}', cond_samples)

# Plot now
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

# Plot the truth

xlb, xub = -3, 3
ylb, yub = -3, 3

grid = jnp.linspace(-3, 3, 1000)
meshgrid = jnp.meshgrid(grid, grid)
cartesian = jnp.dstack(meshgrid)

posterior_pdfs = crescent.pdf_x_cond_y(cartesian, y)
plt.contourf(*meshgrid, posterior_pdfs, cmap=plt.cm.binary)
plt.scatter(cond_samples[:, 0], cond_samples[:, 1], s=1, c='tab:blue', alpha=0.2)

plt.text(1, 1, f'$y={y}$')

plt.grid(linestyle='--', alpha=0.3, which='both')
plt.xlim(xlb, xub)
plt.ylim(ylb, yub)
plt.title('Conditional samples by Feynman--Kac')

plt.tight_layout(pad=0.1)
plt.show()

plt.plot(esss)
plt.show()
