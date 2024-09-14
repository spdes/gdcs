"""
Conditional Schrödinger bridge sampling for the Crescent distribution. Run this after training the DSB model.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from gdcs.target import Crescent
from gdcs.nns import make_st_nn, CrescentMLP
from gdcs.utils import euler_maruyama
from functools import partial

jax.config.update("jax_enable_x64", False)
key = jax.random.PRNGKey(666)

# Define the data
crescent = Crescent(c=1., xi=0.5)

# Load the DSB model for pi_X
# Define the parametric neural network
nn_dt = 1. / 200
key, subkey = jax.random.split(key)
my_nn = CrescentMLP(dt=nn_dt, dim_out=2)
_, _, nn_drift = make_st_nn(subkey, neural_network=my_nn, dim_in=(3,), batch_size=2)
param_bwd = np.load('checkpoints/dsb-xy-15.npz')['param_bwd']

# Times
T = 1.
nsteps = 128
dt = T / nsteps
ts = jnp.linspace(0., T, nsteps + 1)


def ref_sampler(key_, n: int = 1):
    """The reference distribution is a unit Normal.
    """
    return jax.random.normal(key_, shape=(n, 3))


def cond_ref_sampler(key_, n: int = 1):
    """The posterior reference distribution is a unit Normal.
    """
    return jax.random.normal(key_, shape=(n, 2))


def rev_drift(u, t):
    """The drift of the reverse process.
    """
    return jnp.concatenate([nn_drift(u, T - t, param_bwd),
                            jnp.zeros(1)], axis=-1)


def rev_dispersion(_):
    """The drift of the reverse process.
    """
    return jnp.array([1., 1., 0.])


@partial(jax.vmap, in_axes=[0, 0, None])
def rev_sim(key_, u0, ts_):
    return euler_maruyama(key_, u0, ts_, rev_drift, rev_dispersion, integration_nsteps=10, return_path=False)


# Conditional sampling
y = 5
nsamples = 10000

key, subkey = jax.random.split(key)
cond_ref_samples = cond_ref_sampler(subkey, nsamples)
u0s = jnp.concatenate([cond_ref_samples, y * jnp.ones((nsamples, 1))], axis=-1)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=nsamples)
cond_samples = rev_sim(keys, u0s, ts)
np.save(f'cond_samples_cdsb_{y}', cond_samples)

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
plt.title('Conditional samples by Schrödinger bridge')

plt.tight_layout(pad=0.1)
plt.show()
