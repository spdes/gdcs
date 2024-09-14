"""
Reproduce Figures 1 and 2 in the paper.

You need to have already run the conditional sampling scripts which save the generated samples in files.
"""
import jax
import numpy as np
import matplotlib.pyplot as plt
from gdcs.target import Crescent
from gdcs.plot_tools import AxesGrid

crescent = Crescent(c=1., xi=0.5)

# To avoid hugh-size figures, we subsample the plotting data
subsampling = 5
grid_subsampling = 10


@jax.jit
def posterior_pdf(x_, y_):
    return crescent.pdf_x_cond_y(x_, y_)


def load(y_):
    # The grids used to numerically compute the truth
    if y == -1:
        grid_x1 = np.linspace(-4, 4, 2000)
        grid_x2 = np.linspace(-4, 2, 2000)
    elif y == 2:
        grid_x1 = np.linspace(-4, 4, 2000)
        grid_x2 = np.linspace(-4, 3, 2000)
    elif y == 5:
        grid_x1 = np.linspace(-4, 4, 2000)
        grid_x2 = np.linspace(0, 4, 2000)
    else:
        raise ValueError('y not implemented')
    meshgrid = np.meshgrid(grid_x1, grid_x2)
    cartesian = np.dstack(meshgrid)
    pos_pdf_evals = posterior_pdf(cartesian, y_)
    return (np.load(f'cond_samples_cdsb_{y_}.npy'), np.load(f'cond_samples_fk_{y_}.npy'),
            np.load(f'cond_samples_hmc_{y_}.npy'),
            meshgrid, pos_pdf_evals)


# Plot now
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

# Plot
fig = plt.figure(figsize=(14, 12))
grid = AxesGrid(fig, 111, nrows_ncols=(3, 3),
                axes_pad=0.15,
                label_mode='L',
                share_all=False,
                share_x=True,
                share_y=False,
                cbar_location='right',
                cbar_mode='edge',
                cbar_size='3%',
                cbar_pad='1%')

for row, y in enumerate([-1, 2, 5]):
    cond_samples_cdsb, cond_samples_fk, cond_samples_hmc, meshgrids, posterior_pdfs = load(y)
    cond_samples_cdsb = cond_samples_cdsb[::subsampling]
    cond_samples_fk = cond_samples_fk[::subsampling]
    cond_samples_hmc = cond_samples_hmc[::subsampling]
    meshgrids = (meshgrids[0][::grid_subsampling, ::grid_subsampling],
                 meshgrids[1][::grid_subsampling, ::grid_subsampling])
    posterior_pdfs = posterior_pdfs[::grid_subsampling, ::grid_subsampling]

    for col, cond_samples in enumerate([cond_samples_cdsb, cond_samples_fk, cond_samples_hmc]):
        ind = row * 3 + col
        ct = grid[ind].contourf(*meshgrids, posterior_pdfs, levels=12, cmap=plt.cm.binary)
        grid[ind].scatter(cond_samples[:, 0], cond_samples[:, 1], s=1, c='tab:blue', alpha=0.5)
        if (ind == 2) or (ind == 5) or (ind == 8):
            cbar = grid.cbar_axes[ind // 3].colorbar(ct)
            for cb_l in cbar.ax.get_yticklabels():
                cb_l.set_fontsize(12)

        grid[ind].grid(linestyle='--', alpha=0.3, which='both')
        grid[ind].set_xlim(-3, 3)
        if row == 0:
            grid[ind].set_ylim(-3, 1)
        elif row == 1:
            grid[ind].set_ylim(-2, 3)
        else:
            grid[ind].set_ylim(0, 4)

grid[0].set_title('CDSB')
grid[1].set_title('Feynman--Kac')
grid[2].set_title('HMC')

grid[0].set_ylabel('$x_2$ ($y=-1$)')
grid[3].set_ylabel('$x_2$ ($y=2$)')
grid[6].set_ylabel('$x_2$ ($y=5$)')

grid[6].set_xlabel('$x_1$')
grid[7].set_xlabel('$x_1$')
grid[8].set_xlabel('$x_1$')

plt.tight_layout(pad=0.1)
plt.savefig('cond-samples.pdf', transparent=True)
plt.show()

# Plot histograms
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey='row')

y = 5
cond_samples_cdsb, cond_samples_fk, cond_samples_hmc, meshgrids, posterior_pdfs = load(y)
x1_coors = meshgrids[0][0, :]
x2_coors = meshgrids[1][:, 0]
posterior_x1 = np.trapezoid(posterior_pdfs, x=x2_coors, axis=0)
posterior_x2 = np.trapezoid(posterior_pdfs, x=x1_coors, axis=1)

# The hist for marginal x1
for ax in axes[0, :]:
    ax.plot(x1_coors, posterior_x1, linewidth=2, color='black', label='Truth')
    ax.set_xlabel('$x_1$')
axes[0, 0].hist(cond_samples_cdsb[:, 0], bins=50, density=True, alpha=0.3, color='black', label='Sample histogram')
axes[0, 0].set_title('CDSB')
axes[0, 1].hist(cond_samples_fk[:, 0], bins=50, density=True, alpha=0.3, color='black', label='Sample histogram')
axes[0, 1].set_title('Feynman--Kac')
axes[0, 2].hist(cond_samples_hmc[:, 0], bins=50, density=True, alpha=0.3, color='black', label='Sample histogram')
axes[0, 2].set_title('HMC')

# The hist for marginal x2
for ax in axes[1, :]:
    ax.plot(x2_coors, posterior_x2, linewidth=2, color='black', label='Truth')
    ax.set_xlabel('$x_2$')
axes[1, 0].hist(cond_samples_cdsb[:, 1], bins=50, density=True, alpha=0.3, color='black', label='Sample histogram')
axes[1, 1].hist(cond_samples_fk[:, 1], bins=50, density=True, alpha=0.3, color='black', label='Sample histogram')
axes[1, 2].hist(cond_samples_hmc[:, 1], bins=50, density=True, alpha=0.3, color='black', label='Sample histogram')

# Others
for ax in axes.ravel():
    ax.grid(linestyle='--', alpha=0.3, which='both')

axes[0, 0].set_ylabel(f'Probability density')
axes[1, 0].set_ylabel(f'Probability density')
axes[1, 0].legend(loc='upper left')

plt.tight_layout(pad=0.1)
plt.savefig('cond-hist.pdf', transparent=True)
plt.show()
