"""NVIDIA Warp backend for the NBK stochastic IMU error model.

CUDA-only. Uses @wp.kernel for thread-per-environment parallelism.
For AMD/ROCm, use the JAX backend instead.

Usage:
    from imu_nbk.warp_backend import ImuNbkWarp

    sim = ImuNbkWarp(params, n_envs=1024, seed=42)
    noise = sim.step()  # returns wp.array shape (n_envs, 6)
"""

try:
    import warp as wp
except ImportError:
    raise ImportError(
        "NVIDIA Warp not installed. Install with: pip install warp-lang\n"
        "Note: Warp requires NVIDIA CUDA GPU. For AMD/ROCm, use the JAX backend."
    )

import numpy as np
import math

from .params import ImuNbkParams

wp.init()


@wp.struct
class NbkAxisParams:
    N: float
    B: float
    K: float
    corr_time: float


@wp.struct
class NbkParams:
    acc: wp.array(dtype=NbkAxisParams)   # length 3 (per axis)
    gyro: wp.array(dtype=NbkAxisParams)  # length 3 (per axis)
    dt: float


@wp.kernel
def _nbk_step_kernel(
    z_B: wp.array2d(dtype=float),       # (n_envs, 6): acc[3] + gyro[3]
    z_K: wp.array2d(dtype=float),       # (n_envs, 6)
    output: wp.array2d(dtype=float),    # (n_envs, 6)
    acc_N: wp.array(dtype=float),       # (3,)
    acc_B: wp.array(dtype=float),       # (3,)
    acc_K: wp.array(dtype=float),       # (3,)
    acc_corr_time: wp.array(dtype=float),  # (3,)
    gyro_N: wp.array(dtype=float),
    gyro_B: wp.array(dtype=float),
    gyro_K: wp.array(dtype=float),
    gyro_corr_time: wp.array(dtype=float),
    dt: float,
    rng_states: wp.array(dtype=wp.uint32),
):
    env_id = wp.tid()
    state = rng_states[env_id]

    for ax in range(3):
        N = acc_N[ax]
        B = acc_B[ax]
        K = acc_K[ax]
        corr_time = acc_corr_time[ax]

        S_N = N * N
        S_B = (2.0 * B * B * 0.6931471805599453) / (3.141592653589793 * 0.4365 * 0.4365 * corr_time)
        S_K = K * K
        beta = 1.0 / corr_time

        phi = wp.exp(-beta * dt)
        Q_B = S_B / (2.0 * beta) * (1.0 - wp.exp(-2.0 * beta * dt))
        Q_K = S_K * dt
        Q_N = S_N / dt

        r0 = wp.randf(state)
        state = state + wp.uint32(1)
        r1 = wp.randf(state)
        state = state + wp.uint32(1)
        r2 = wp.randf(state)
        state = state + wp.uint32(1)

        # Box-Muller for normal distribution
        u1 = r0
        u2 = r1
        if u1 < 1.0e-10:
            u1 = 1.0e-10
        mag = wp.sqrt(-2.0 * wp.log(u1))
        n0 = mag * wp.cos(2.0 * 3.141592653589793 * u2)

        u1 = r2
        state_tmp = state
        r3 = wp.randf(state_tmp)
        state = state_tmp + wp.uint32(1)
        if u1 < 1.0e-10:
            u1 = 1.0e-10
        mag2 = wp.sqrt(-2.0 * wp.log(u1))
        n1 = mag2 * wp.cos(2.0 * 3.141592653589793 * r3)

        r4 = wp.randf(state)
        state = state + wp.uint32(1)
        r5 = wp.randf(state)
        state = state + wp.uint32(1)
        if r4 < 1.0e-10:
            r4 = 1.0e-10
        mag3 = wp.sqrt(-2.0 * wp.log(r4))
        n2 = mag3 * wp.cos(2.0 * 3.141592653589793 * r5)

        w_N = wp.sqrt(Q_N) * n0
        w_B = wp.sqrt(Q_B) * n1
        w_K = wp.sqrt(Q_K) * n2

        cur_z_B = z_B[env_id, ax]
        cur_z_K = z_K[env_id, ax]

        new_z_B = phi * cur_z_B + w_B
        new_z_K = cur_z_K + w_K
        out = cur_z_B + cur_z_K + w_N

        z_B[env_id, ax] = new_z_B
        z_K[env_id, ax] = new_z_K
        output[env_id, ax] = out

    for ax in range(3):
        N = gyro_N[ax]
        B = gyro_B[ax]
        K = gyro_K[ax]
        corr_time = gyro_corr_time[ax]

        S_N = N * N
        S_B = (2.0 * B * B * 0.6931471805599453) / (3.141592653589793 * 0.4365 * 0.4365 * corr_time)
        S_K = K * K
        beta = 1.0 / corr_time

        phi = wp.exp(-beta * dt)
        Q_B = S_B / (2.0 * beta) * (1.0 - wp.exp(-2.0 * beta * dt))
        Q_K = S_K * dt
        Q_N = S_N / dt

        r0 = wp.randf(state)
        state = state + wp.uint32(1)
        r1 = wp.randf(state)
        state = state + wp.uint32(1)
        r2 = wp.randf(state)
        state = state + wp.uint32(1)

        if r0 < 1.0e-10:
            r0 = 1.0e-10
        mag = wp.sqrt(-2.0 * wp.log(r0))
        n0 = mag * wp.cos(2.0 * 3.141592653589793 * r1)

        r3 = wp.randf(state)
        state = state + wp.uint32(1)
        r4 = wp.randf(state)
        state = state + wp.uint32(1)
        if r2 < 1.0e-10:
            r2 = 1.0e-10
        mag2 = wp.sqrt(-2.0 * wp.log(r2))
        n1 = mag2 * wp.cos(2.0 * 3.141592653589793 * r3)

        r5 = wp.randf(state)
        state = state + wp.uint32(1)
        if r4 < 1.0e-10:
            r4 = 1.0e-10
        mag3 = wp.sqrt(-2.0 * wp.log(r4))
        n2 = mag3 * wp.cos(2.0 * 3.141592653589793 * r5)

        w_N = wp.sqrt(Q_N) * n0
        w_B = wp.sqrt(Q_B) * n1
        w_K = wp.sqrt(Q_K) * n2

        idx = 3 + ax
        cur_z_B = z_B[env_id, idx]
        cur_z_K = z_K[env_id, idx]

        new_z_B = phi * cur_z_B + w_B
        new_z_K = cur_z_K + w_K
        out = cur_z_B + cur_z_K + w_N

        z_B[env_id, idx] = new_z_B
        z_K[env_id, idx] = new_z_K
        output[env_id, idx] = out

    rng_states[env_id] = state


class ImuNbkWarp:
    """Warp-based batched IMU NBK simulator.

    Manages device state and launches CUDA kernels per timestep.
    """

    def __init__(self, params: ImuNbkParams, n_envs: int = 1, seed: int = 42,
                 device: str = "cuda:0"):
        self.n_envs = n_envs
        self.dt = params.dt
        self.device = device

        # State arrays on device
        self.z_B = wp.zeros((n_envs, 6), dtype=float, device=device)
        self.z_K = wp.zeros((n_envs, 6), dtype=float, device=device)
        self.output = wp.zeros((n_envs, 6), dtype=float, device=device)

        # Parameter arrays on device
        self.acc_N = wp.array(params.acc_N, dtype=float, device=device)
        self.acc_B = wp.array(params.acc_B, dtype=float, device=device)
        self.acc_K = wp.array(params.acc_K, dtype=float, device=device)
        self.acc_corr_time = wp.array(params.acc_corr_time, dtype=float, device=device)
        self.gyro_N = wp.array(params.gyro_N, dtype=float, device=device)
        self.gyro_B = wp.array(params.gyro_B, dtype=float, device=device)
        self.gyro_K = wp.array(params.gyro_K, dtype=float, device=device)
        self.gyro_corr_time = wp.array(params.gyro_corr_time, dtype=float, device=device)

        # Per-environment RNG seeds
        seeds = np.arange(seed, seed + n_envs, dtype=np.uint32)
        self.rng_states = wp.array(seeds, dtype=wp.uint32, device=device)

    def step(self) -> wp.array:
        """Run one timestep for all environments.

        Returns:
            wp.array shape (n_envs, 6) with stochastic noise output.
        """
        wp.launch(
            _nbk_step_kernel,
            dim=self.n_envs,
            inputs=[
                self.z_B, self.z_K, self.output,
                self.acc_N, self.acc_B, self.acc_K, self.acc_corr_time,
                self.gyro_N, self.gyro_B, self.gyro_K, self.gyro_corr_time,
                self.dt, self.rng_states,
            ],
            device=self.device,
        )
        return self.output

    def reset(self):
        """Reset all state to zero."""
        self.z_B.zero_()
        self.z_K.zero_()
        self.output.zero_()

    def get_output_numpy(self) -> np.ndarray:
        """Copy latest output to host as NumPy array."""
        return self.output.numpy()
