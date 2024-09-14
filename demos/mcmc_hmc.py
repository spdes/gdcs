"""
MALA for the Crescent distribution.
"""
import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import matplotlib.pyplot as plt
from gdcs.target import Crescent

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)

# Define the data
crescent = Crescent(c=1., xi=0.5)


def log_pdf(x_):
    return crescent.logpdf_y_cond_x(y, x_) + crescent.logpdf_x(x_)


# Conditional sampling
y = -1
nsamples = 10000
nburnin = 200

dt = 0.35
inv_mass = jnp.ones(2)
x = jnp.zeros(2)

hmc_obj = blackjax.hmc(log_pdf, dt, inv_mass, 100)
sampler = jax.jit(hmc_obj.step)
state = hmc_obj.init(x)

cond_samples = np.zeros((nsamples + nburnin, 2))

for i in range(nsamples + nburnin):
    key, subkey = jax.random.split(key)
    state, _ = sampler(subkey, state)
    cond_samples[i] = state.position
    print(i)

cond_samples = cond_samples[nburnin:]
np.save(f'cond_samples_hmc_{y}', cond_samples)

# Plot now
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

# Plot the truth
xlb, xub = -3, 3
ylb, yub = -3, 3
grid = jnp.linspace(-4, 4, 1000)
meshgrid = jnp.meshgrid(grid, grid)
cartesian = jnp.dstack(meshgrid)

posterior_pdfs = crescent.pdf_x_cond_y(cartesian, y)
plt.contourf(*meshgrid, posterior_pdfs, cmap=plt.cm.binary)
plt.scatter(cond_samples[:, 0], cond_samples[:, 1], s=1, c='tab:blue', alpha=0.2)

plt.text(1, 1, f'$y={y}$')

plt.grid(linestyle='--', alpha=0.3, which='both')
plt.xlim(xlb, xub)
plt.ylim(ylb, yub)

plt.tight_layout(pad=0.1)
plt.show()

plt.plot(cond_samples[:, 0])
plt.show()
