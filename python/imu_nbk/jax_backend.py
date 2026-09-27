"""JAX backend for the NBK stochastic IMU error model.

Designed for batched MJX training: jit-compiled, vmap-compatible, functional RNG.
Supports CUDA, ROCm, TPU, and CPU via JAX's backend abstraction.

Usage with MJX:
    from imu_nbk import load_config
    from imu_nbk.jax_backend import init_state, step, batched_step, params_to_jax

    params = load_config("config/adis16607.yaml")
    jax_params = params_to_jax(params)
    state = init_state()  # per-environment

    # In training loop (after mjx.step):
    state, noise, rng = step(state, jax_params, rng_key)
    # Or batched across N environments:
    states, noises, rngs = batched_step(states, jax_params, rng_keys)
"""

import jax
import jax.numpy as jnp
from jax import random

from .params import ImuNbkParams


def params_to_jax(params: ImuNbkParams) -> dict:
    """Convert ImuNbkParams to a JAX-compatible dict of arrays.

    Returns a dict (valid JAX pytree) with float64 arrays for all parameters.
    Pass this as a static argument or use jax.tree_util for tracing.
    """
    return {
        "acc_N": jnp.array(params.acc_N),
        "acc_B": jnp.array(params.acc_B),
        "acc_K": jnp.array(params.acc_K),
        "acc_corr_time": jnp.array(params.acc_corr_time),
        "gyro_N": jnp.array(params.gyro_N),
        "gyro_B": jnp.array(params.gyro_B),
        "gyro_K": jnp.array(params.gyro_K),
        "gyro_corr_time": jnp.array(params.gyro_corr_time),
        "dt": jnp.float64(params.dt),
    }


def init_state() -> dict:
    """Create zero-initialized Gauss-Markov state (JAX arrays).

    Returns:
        Pytree with keys: acc_z_B, acc_z_K, gyro_z_B, gyro_z_K — each shape (3,).
    """
    return {
        "acc_z_B": jnp.zeros(3),
        "acc_z_K": jnp.zeros(3),
        "gyro_z_B": jnp.zeros(3),
        "gyro_z_K": jnp.zeros(3),
    }


def _nbk_axis_step(z_B, z_K, N, B, K, corr_time, dt, key):
    """Single-axis NBK update — matches C++ imu_simulator.cpp:973-1046."""
    S_N = N * N
    S_B = (2.0 * B**2 * jnp.log(2.0)) / (jnp.pi * 0.4365**2 * corr_time)
    S_K = K * K
    beta = 1.0 / corr_time

    phi = jnp.exp(-beta * dt)
    Q_B = S_B / (2.0 * beta) * (1.0 - jnp.exp(-2.0 * beta * dt))
    Q_K = S_K * dt
    Q_N = S_N / dt

    k1, k2, k3 = random.split(key, 3)
    w_N = jnp.sqrt(Q_N) * random.normal(k1)
    w_B = jnp.sqrt(Q_B) * random.normal(k2)
    w_K = jnp.sqrt(Q_K) * random.normal(k3)

    z_B_new = phi * z_B + w_B
    z_K_new = z_K + w_K
    output = z_B + z_K + w_N

    return z_B_new, z_K_new, output


@jax.jit
def step(state: dict, params: dict, rng_key) -> tuple:
    """One timestep of the full NBK model (6 axes).

    Args:
        state: Pytree from init_state().
        params: Dict from params_to_jax().
        rng_key: JAX PRNG key.

    Returns:
        (new_state, noise[6], new_rng_key)
        noise order: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    """
    dt = params["dt"]

    # Split into 18 sub-keys (3 per axis × 6 axes) + 1 carry
    keys = random.split(rng_key, 19)
    carry_key = keys[0]
    axis_keys = keys[1:]  # shape (18,)

    noise = jnp.zeros(6)
    new_acc_z_B = jnp.zeros(3)
    new_acc_z_K = jnp.zeros(3)
    new_gyro_z_B = jnp.zeros(3)
    new_gyro_z_K = jnp.zeros(3)

    # Accelerometer axes
    for ax in range(3):
        ax_key = axis_keys[ax]
        z_B_new, z_K_new, out = _nbk_axis_step(
            state["acc_z_B"][ax], state["acc_z_K"][ax],
            params["acc_N"][ax], params["acc_B"][ax], params["acc_K"][ax],
            params["acc_corr_time"][ax], dt, ax_key)
        new_acc_z_B = new_acc_z_B.at[ax].set(z_B_new)
        new_acc_z_K = new_acc_z_K.at[ax].set(z_K_new)
        noise = noise.at[ax].set(out)

    # Gyroscope axes
    for ax in range(3):
        ax_key = axis_keys[3 + ax]
        z_B_new, z_K_new, out = _nbk_axis_step(
            state["gyro_z_B"][ax], state["gyro_z_K"][ax],
            params["gyro_N"][ax], params["gyro_B"][ax], params["gyro_K"][ax],
            params["gyro_corr_time"][ax], dt, ax_key)
        new_gyro_z_B = new_gyro_z_B.at[ax].set(z_B_new)
        new_gyro_z_K = new_gyro_z_K.at[ax].set(z_K_new)
        noise = noise.at[3 + ax].set(out)

    new_state = {
        "acc_z_B": new_acc_z_B,
        "acc_z_K": new_acc_z_K,
        "gyro_z_B": new_gyro_z_B,
        "gyro_z_K": new_gyro_z_K,
    }

    return new_state, noise, carry_key


# Vectorized version for batched environments
batched_step = jax.vmap(step, in_axes=(0, None, 0))


def init_state_batched(n_envs: int) -> dict:
    """Create batched zero-initialized state for N parallel environments.

    Returns:
        Pytree with shape (n_envs, 3) for each state array.
    """
    return {
        "acc_z_B": jnp.zeros((n_envs, 3)),
        "acc_z_K": jnp.zeros((n_envs, 3)),
        "gyro_z_B": jnp.zeros((n_envs, 3)),
        "gyro_z_K": jnp.zeros((n_envs, 3)),
    }


def run_static_simulation(params: ImuNbkParams, duration_s: float,
                          seed: int = 42) -> "jnp.ndarray":
    """Run a static simulation using JAX (for Allan variance validation).

    Args:
        params: ImuNbkParams from YAML.
        duration_s: Duration in seconds.
        seed: RNG seed.

    Returns:
        JAX array shape (n_samples, 6).
    """
    jax_params = params_to_jax(params)
    n_samples = int(duration_s / params.dt)
    state = init_state()
    rng = random.PRNGKey(seed)

    def scan_fn(carry, _):
        st, key = carry
        st, noise, key = step(st, jax_params, key)
        return (st, key), noise

    (_, _), noises = jax.lax.scan(scan_fn, (state, rng), None, length=n_samples)
    return noises
