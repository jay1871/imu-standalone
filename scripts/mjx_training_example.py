#!/usr/bin/env python3
"""Example: MJX batched RL training loop with realistic IMU noise.

This shows the pattern for integrating the IMU error model into a
JAX-based reinforcement learning pipeline using MuJoCo MJX.

The key idea:
  1. MJX steps physics on GPU (no plugin support in MJX)
  2. MJX provides noiseless accel/gyro via built-in sensors
  3. imu_nbk JAX backend adds realistic NBK noise on GPU
  4. Your policy sees the noisy IMU — trains to be robust

Requirements:
  pip install mujoco mujoco-mjx "jax[cuda12]"
  pip install -e ../python  (the imu_nbk package)
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from imu_nbk import load_config
from imu_nbk.jax_backend import (
    init_state_batched,
    batched_step,
    params_to_jax,
)

# --- Configuration ---
N_ENVS = 1024
EPISODE_LENGTH = 1000  # steps
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "free_body_imu.xml"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "adis16607.yaml"


def make_env(model_path: Path, n_envs: int):
    """Load MuJoCo model and replicate across environments."""
    m = mujoco.MjModel.from_xml_path(str(model_path))
    mx = mjx.put_model(m)

    d = mujoco.MjData(m)
    dx = mjx.put_data(m, d)
    dx_batch = jax.vmap(lambda _: dx)(jnp.arange(n_envs))

    # Find noiseless sensor addresses
    acc_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "acc_true")]
    gyro_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro_true")]

    return mx, dx_batch, acc_adr, gyro_adr


def main():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Environments: {N_ENVS}")

    # 1. Load physics model
    mx, dx_batch, acc_adr, gyro_adr = make_env(MODEL_PATH, N_ENVS)

    # 2. Load IMU error model
    imu_params = params_to_jax(load_config(str(CONFIG_PATH)))
    imu_states = init_state_batched(N_ENVS)
    rng_keys = jax.random.split(jax.random.PRNGKey(0), N_ENVS)

    # 3. JIT-compile the combined step
    @jax.jit
    def env_step(dx_batch, imu_states, rng_keys):
        # Physics step (all envs on GPU)
        dx_batch = jax.vmap(lambda dx: mjx.step(mx, dx))(dx_batch)

        # Read noiseless sensors
        true_acc = dx_batch.sensordata[:, acc_adr:acc_adr + 3]
        true_gyro = dx_batch.sensordata[:, gyro_adr:gyro_adr + 3]

        # Apply NBK stochastic error model (all envs on GPU)
        imu_states, noises, rng_keys = batched_step(imu_states, imu_params, rng_keys)

        # What your policy observes
        obs_acc = true_acc + noises[:, :3]
        obs_gyro = true_gyro + noises[:, 3:]

        return dx_batch, imu_states, rng_keys, obs_acc, obs_gyro

    # 4. Run one episode
    print(f"\nRunning {EPISODE_LENGTH} steps across {N_ENVS} environments...")

    for step_i in range(EPISODE_LENGTH):
        dx_batch, imu_states, rng_keys, obs_acc, obs_gyro = env_step(
            dx_batch, imu_states, rng_keys)

        # --- Your RL policy goes here ---
        # obs = jnp.concatenate([obs_acc, obs_gyro, ...], axis=-1)
        # action = policy(obs)
        # reward = compute_reward(dx_batch, action)
        # ...

        if step_i == 0:
            # Print first step to verify shapes
            jax.block_until_ready(obs_acc)
            print(f"\n  Step 0 shapes:")
            print(f"    obs_acc:  {obs_acc.shape}  (n_envs, 3)")
            print(f"    obs_gyro: {obs_gyro.shape}  (n_envs, 3)")
            print(f"\n  Env 0, step 0 readings:")
            print(f"    Accel (m/s²): {obs_acc[0]}")
            print(f"    Gyro (rad/s): {obs_gyro[0]}")

    jax.block_until_ready(obs_acc)
    print(f"\n  Done. {N_ENVS} x {EPISODE_LENGTH} = {N_ENVS * EPISODE_LENGTH:,} total steps")
    print("  All on GPU with realistic IMU noise — ready for policy training.")


if __name__ == "__main__":
    main()
