"""
Train a (conditional) dynamic Schrödinger bridge (DSB) targeting at the prior model pi_X or the joint pi_{X, Y}.
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from gdcs.target import Crescent
from gdcs.nns import make_st_nn, CrescentMLP
from gdcs.dsb import ipf_loss_cont as ipf_loss
from gdcs.utils import euler_maruyama
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='x', help='Train DSB targeting at X or jointly X and Y.')
parser.add_argument('--nsbs', type=int, default=10, help='The number of DSB iterations.')
args = parser.parse_args()

jax.config.update("jax_enable_x64", False)
key = jax.random.PRNGKey(666)

# Define the data
crescent = Crescent(c=1., xi=0.5)


# Sampler for p(x)
@partial(jax.vmap, in_axes=[0])
def sampler_x(key_):
    return crescent.sampler_x(key_)


# Sampler for p(x, y)
@partial(jax.vmap, in_axes=[0])
def sampler_xy(key_):
    x_, y_ = crescent.sampler_joint(key_)
    return jnp.concatenate([x_, y_[..., None]], axis=-1)


# Define the DSB
target = args.target
T = 1.  # End time
sampler_data = sampler_x if target == 'x' else sampler_xy


def reference_drift(x, t, p):
    """The reference is a Brownian motion.
    When use for conditional sampling, the SDE for the y part is dY(t)=0, hence no drift.
    """
    return -0.5 * x if target == 'x' else jnp.array([-0.5, -0.5, 0.]) * x


def reference_dispersion(t):
    return 1. if target == 'x' else jnp.array([1., 1., 0.])


def ref_sampler(key_, n: int = 1):
    """The reference distribution is a unit Normal.
    """
    return jax.random.normal(key_, shape=(n, 2 if target == 'x' else 3))


# Define the training parameters
train_nsamples = 1024  # The batch sample size
train_nsteps = 128  # The number of time steps
nn_dt = 1. / 200  # This doesn't need to be the exact dt, merely a scale factor.
nepochs = 5  # The number of epochs for each DSB iteration
nsbs = args.nsbs  # The number of DSB iterations

# Create the neural network instance
key, subkey = jax.random.split(key)
my_nn = CrescentMLP(dt=nn_dt, dim_out=2)
param_fwd, _, nn_fn = make_st_nn(subkey, neural_network=my_nn,
                                 dim_in=(2 if target == 'x' else 3,), batch_size=train_nsamples)
param_bwd, _, _ = make_st_nn(subkey, neural_network=my_nn,
                             dim_in=(2 if target == 'x' else 3,), batch_size=train_nsamples)


def nn_drift(x, t, p):
    return nn_fn(x, t, p) if target == 'x' else jnp.concatenate([nn_fn(x, t, p),
                                                                 jnp.zeros((*x.shape[:-1], 1))], axis=-1)


# Set up Optax
lr = 1e-4
niters_per_epoch = 100
schedule = 'const'
if schedule == 'cos':
    until_steps = int(0.95 * nepochs) * niters_per_epoch
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=until_steps, alpha=1e-2)
elif schedule == 'exp':
    schedule = optax.exponential_decay(lr, niters_per_epoch, .96)
else:
    schedule = optax.constant_schedule(lr)

optimiser = optax.adam(learning_rate=schedule)
optimiser = optax.chain(optax.clip_by_global_norm(1.),
                        optimiser)


# Make loss functions

def loss_fn_init(param_bwd_, param_fwd_, key_, data_samples):
    """Simulate the forward data -> sth. to learn its backward.
    This loss is for the first iteration that uses the reference SDE.
    """
    key_loss, key_ts = jax.random.split(key_, num=2)
    rnd_ts = jnp.hstack([0.,
                         jnp.sort(jax.random.uniform(key_ts, (train_nsteps - 1,), minval=0. + 1e-5, maxval=T - 1e-5)),
                         T])
    return ipf_loss(key_loss, param_bwd_, param_fwd_, data_samples, rnd_ts, nn_drift,
                    reference_drift, reference_dispersion)


def loss_fn_bwd(param_bwd_, param_fwd_, key_, data_samples):
    """Simulate the forward data -> sth. to learn its backward.
    """
    key_loss, key_ts = jax.random.split(key_, num=2)
    rnd_ts = jnp.hstack([0.,
                         jnp.sort(jax.random.uniform(key_ts, (train_nsteps - 1,), minval=0. + 1e-5, maxval=T - 1e-5)),
                         T])
    return ipf_loss(key_loss, param_bwd_, param_fwd_, data_samples, rnd_ts, nn_drift, nn_drift, reference_dispersion)


def loss_fn_fwd(param_fwd_, param_bwd_, key_, ref_samples):
    """Simulate the backward sth. <- ref to learn its forward.
    """
    key_loss, key_ts = jax.random.split(key_, num=2)
    rnd_ts = jnp.hstack([0.,
                         jnp.sort(jax.random.uniform(key_ts, (train_nsteps - 1,), minval=0. + 1e-5, maxval=T)),
                         T])
    return ipf_loss(key_loss, param_fwd_, param_bwd_, ref_samples, T - rnd_ts, nn_drift, nn_drift, reference_dispersion)


# Make optax kernels
@jax.jit
def optax_kernel_init(param_bwd_, opt_state_, param_fwd_, key_, data_samples):
    loss, grad = jax.value_and_grad(loss_fn_init)(param_bwd_, param_fwd_, key_, data_samples)
    updates, opt_state_ = optimiser.update(grad, opt_state_, param_bwd_)
    param_bwd_ = optax.apply_updates(param_bwd_, updates)
    return param_bwd_, opt_state_, loss


@jax.jit
def optax_kernel_bwd(param_bwd_, opt_state_, param_fwd_, key_, data_samples):
    loss, grad = jax.value_and_grad(loss_fn_bwd)(param_bwd_, param_fwd_, key_, data_samples)
    updates, opt_state_ = optimiser.update(grad, opt_state_, param_bwd_)
    param_bwd_ = optax.apply_updates(param_bwd_, updates)
    return param_bwd_, opt_state_, loss


@jax.jit
def optax_kernel_fwd(param_fwd_, opt_state_, param_bwd_, key_, ref_samples):
    loss, grad = jax.value_and_grad(loss_fn_fwd)(param_fwd_, param_bwd_, key_, ref_samples)
    updates, opt_state_ = optimiser.update(grad, opt_state_, param_fwd_)
    param_fwd_ = optax.apply_updates(param_fwd_, updates)
    return param_fwd_, opt_state_, loss


# Make a loop for Schrodinger bridge iteration
def sb_kernel(param_fwd_, param_bwd_, opt_state_fwd_, opt_state_bwd_, key_train, sb_step: int):
    # Learn the backward process, i.e., data <- ref
    for i in range(nepochs):
        key_train, subkey = jax.random.split(key_train)
        for j in range(niters_per_epoch):
            subkey, subkey2 = jax.random.split(subkey)
            x0s = sampler_data(jax.random.split(subkey, num=train_nsamples))
            if sb_step == 0:
                param_bwd_, opt_state_bwd_, loss = optax_kernel_init(param_bwd_, opt_state_bwd_, param_fwd_, subkey2,
                                                                     x0s)
            else:
                param_bwd_, opt_state_bwd_, loss = optax_kernel_bwd(param_bwd_, opt_state_bwd_, param_fwd_, subkey2,
                                                                    x0s)
            print(f'Learning bwd | SB iter: {sb_step} | Epoch: {i} / {nepochs}, '
                  f'iter: {j} / {niters_per_epoch} | loss: {loss:.4f}')

    # Learn the forward process, i.e., data -> ref
    for i in range(nepochs):
        key_train, subkey = jax.random.split(key_train)
        for j in range(niters_per_epoch):
            subkey, subkey2 = jax.random.split(subkey)
            xTs = ref_sampler(subkey, n=train_nsamples)
            param_fwd_, opt_state_fwd_, loss = optax_kernel_fwd(param_fwd_, opt_state_fwd_, param_bwd_, subkey2, xTs)
            print(f'Learning fwd | SB iter: {sb_step} | Epoch: {i} / {nepochs}, '
                  f'iter: {j} / {niters_per_epoch} | loss: {loss:.4f}')

    return param_fwd_, param_bwd_, opt_state_fwd_, opt_state_bwd_


# Now the Scrodinger bridge loop
key_sb, _ = jax.random.split(key)
opt_state_fwd = optimiser.init(param_fwd)
opt_state_bwd = optimiser.init(param_bwd)

for sb_iter in range(nsbs):
    key_sb, subkey = jax.random.split(key_sb)
    param_fwd, param_bwd, opt_state_fwd, opt_state_bwd = sb_kernel(param_fwd, param_bwd, opt_state_fwd, opt_state_bwd,
                                                                   subkey, sb_iter)
    np.savez(f'./checkpoints/dsb-{target}-{sb_iter}', param_fwd=param_fwd, param_bwd=param_bwd)

# Verify if the Schrodinger bridge is learnt properly
nsteps = 1024
ts = jnp.linspace(0., T, nsteps)


@jax.jit
@partial(jax.vmap, in_axes=[0, 0])
def rev_sim(key_, xT):
    def rev_drift(x, t):
        return nn_drift(x, t, param_bwd)

    return euler_maruyama(key_, xT, T - ts, rev_drift, reference_dispersion, integration_nsteps=10, return_path=False)


@jax.jit
@partial(jax.vmap, in_axes=[0, 0])
def fwd_sim(key_, x0):
    def fwd_drift(x, t):
        return nn_drift(x, t, param_fwd)

    return euler_maruyama(key_, x0, ts, fwd_drift, reference_dispersion, integration_nsteps=10, return_path=False)


key_test, _ = jax.random.split(key)
key_test_fwd, key_test_bwd = jax.random.split(key_test)

keys = jax.random.split(key_test_fwd, num=5000)
x0s = sampler_data(keys)
approx_ref_samples = fwd_sim(keys, x0s)

keys = jax.random.split(key_test_bwd, num=5000)
ref_samples = ref_sampler(key_test_bwd, n=5000)
approx_x0s = rev_sim(keys, ref_samples)

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
axes[0].scatter(x0s[:, 0], x0s[:, 1], s=1, alpha=0.5)
axes[0].scatter(approx_x0s[:, 0], approx_x0s[:, 1], s=1, alpha=0.5)

axes[1].scatter(ref_samples[:, 0], ref_samples[:, 1], s=1, alpha=0.5)
axes[1].scatter(approx_ref_samples[:, 0], approx_ref_samples[:, 1], s=1, alpha=0.5)

plt.tight_layout(pad=0.1)
plt.show()
