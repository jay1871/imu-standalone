"""NumPy reference implementation of the NBK stochastic IMU error model.

This mirrors the C++ ImuSimulator::updateDiscreteTimeEquivalentStateSpaceModel()
exactly (src/imu_simulator.cpp:973-1046). It serves as the cross-backend
verification reference.

The API is functional: state in, state out. No mutable class instances.
"""

import numpy as np
from numpy.random import Generator

from .params import ImuNbkParams


def init_state() -> dict:
    """Create zero-initialized Gauss-Markov state.

    Returns:
        State dict with keys: acc_z_B, acc_z_K, gyro_z_B, gyro_z_K
        Each is shape (3,) — one value per axis.
    """
    return {
        "acc_z_B": np.zeros(3),
        "acc_z_K": np.zeros(3),
        "gyro_z_B": np.zeros(3),
        "gyro_z_K": np.zeros(3),
    }


def _nbk_axis_update(z_B: float, z_K: float, N: float, B: float, K: float,
                     corr_time: float, dt: float, rng: Generator,
                     fixed_draws: np.ndarray | None = None,
                     z_B_poles: np.ndarray | None = None,
                     B_poles: np.ndarray | None = None,
                     corr_time_poles: np.ndarray | None = None) -> tuple:
    """Single-axis NBK stochastic update (matches C++ exactly, single- or multi-pole)."""
    S_N = N * N
    S_K = K * K
    Q_K = S_K * dt
    Q_N = S_N / dt

    if B_poles is not None and corr_time_poles is not None and fixed_draws is None:
        n_poles = len(B_poles)
        r = rng.standard_normal(n_poles + 2)
        w_N = np.sqrt(Q_N) * r[0]
        w_K = np.sqrt(Q_K) * r[-1]
        cur_poles = z_B_poles if z_B_poles is not None else np.zeros(n_poles)
        beta_p = 1.0 / corr_time_poles
        S_B_p = (2.0 * B_poles**2 * np.log(2.0)) / (np.pi * 0.4365**2 * corr_time_poles)
        phi_p = np.exp(-beta_p * dt)
        Q_B_p = S_B_p / (2.0 * beta_p) * (1.0 - np.exp(-2.0 * beta_p * dt))
        new_poles = phi_p * cur_poles + np.sqrt(Q_B_p) * r[1:1 + n_poles]
        z_B_new = float(np.sum(new_poles))
        z_K_new = z_K + w_K
        output = float(np.sum(cur_poles)) + z_K + w_N
        return z_B_new, z_K_new, output, new_poles

    S_B = (2.0 * B * B * np.log(2.0)) / (np.pi * 0.4365**2 * corr_time)
    beta = 1.0 / corr_time

    phi = np.exp(-beta * dt)
    Q_B = S_B / (2.0 * beta) * (1.0 - np.exp(-2.0 * beta * dt))

    if fixed_draws is not None:
        r = fixed_draws
    else:
        r = rng.standard_normal(3)

    w_N = np.sqrt(Q_N) * r[0]
    w_B = np.sqrt(Q_B) * r[1]
    w_K = np.sqrt(Q_K) * r[2]

    # State update
    z_B_new = phi * z_B + w_B
    z_K_new = z_K + w_K

    # Output uses CURRENT state (before update), same as C++
    output = z_B + z_K + w_N

    return z_B_new, z_K_new, output, None


def step(state: dict, params: ImuNbkParams, rng: Generator,
         fixed_draws: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """One timestep of the NBK stochastic model (all 6 axes).

    This produces the stochastic noise component only (no deterministic errors,
    no gravity, no turn-on bias). Matches the output of the C++ stochastic
    error pipeline when all deterministic errors are disabled.

    Args:
        state: Dict from init_state() with current Gauss-Markov states.
        params: ImuNbkParams loaded from YAML.
        rng: NumPy random generator for stochastic draws.
        fixed_draws: Optional (18,) array of fixed random draws for testing.
                     Order: acc_x[3], acc_y[3], acc_z[3], gyro_x[3], gyro_y[3], gyro_z[3]

    Returns:
        (new_state, noise) where noise is shape (6,):
            [acc_noise_x, acc_noise_y, acc_noise_z, gyro_noise_x, gyro_noise_y, gyro_noise_z]
    """
    dt = params.dt
    new_state = {}
    noise = np.zeros(6)

    # Accelerometer (axes 0,1,2)
    new_acc_z_B = np.zeros(3)
    new_acc_z_K = np.zeros(3)
    acc_poles_state = state.get("acc_z_B_poles", None)
    new_acc_poles = (
        np.zeros_like(params.acc_B_poles)
        if params.acc_B_poles is not None and fixed_draws is None
        else None
    )
    for ax in range(3):
        draws = fixed_draws[ax * 3:(ax + 1) * 3] if fixed_draws is not None else None
        z_B_new, z_K_new, out, poles_new = _nbk_axis_update(
            state["acc_z_B"][ax], state["acc_z_K"][ax],
            params.acc_N[ax], params.acc_B[ax], params.acc_K[ax],
            params.acc_corr_time[ax], dt, rng, draws,
            acc_poles_state[ax] if acc_poles_state is not None else None,
            params.acc_B_poles[ax] if params.acc_B_poles is not None else None,
            params.acc_corr_time_poles[ax] if params.acc_corr_time_poles is not None else None,
        )
        new_acc_z_B[ax] = z_B_new
        new_acc_z_K[ax] = z_K_new
        if new_acc_poles is not None and poles_new is not None:
            new_acc_poles[ax] = poles_new
        noise[ax] = out

    # Gyroscope (axes 3,4,5)
    new_gyro_z_B = np.zeros(3)
    new_gyro_z_K = np.zeros(3)
    gyro_poles_state = state.get("gyro_z_B_poles", None)
    new_gyro_poles = (
        np.zeros_like(params.gyro_B_poles)
        if params.gyro_B_poles is not None and fixed_draws is None
        else None
    )
    for ax in range(3):
        draws = fixed_draws[(3 + ax) * 3:(4 + ax) * 3] if fixed_draws is not None else None
        z_B_new, z_K_new, out, poles_new = _nbk_axis_update(
            state["gyro_z_B"][ax], state["gyro_z_K"][ax],
            params.gyro_N[ax], params.gyro_B[ax], params.gyro_K[ax],
            params.gyro_corr_time[ax], dt, rng, draws,
            gyro_poles_state[ax] if gyro_poles_state is not None else None,
            params.gyro_B_poles[ax] if params.gyro_B_poles is not None else None,
            params.gyro_corr_time_poles[ax] if params.gyro_corr_time_poles is not None else None,
        )
        new_gyro_z_B[ax] = z_B_new
        new_gyro_z_K[ax] = z_K_new
        if new_gyro_poles is not None and poles_new is not None:
            new_gyro_poles[ax] = poles_new
        noise[3 + ax] = out

    new_state = {
        "acc_z_B": new_acc_z_B,
        "acc_z_K": new_acc_z_K,
        "gyro_z_B": new_gyro_z_B,
        "gyro_z_K": new_gyro_z_K,
    }
    if new_acc_poles is not None:
        new_state["acc_z_B_poles"] = new_acc_poles
    if new_gyro_poles is not None:
        new_state["gyro_z_B_poles"] = new_gyro_poles

    return new_state, noise


def run_static_simulation(params: ImuNbkParams, duration_s: float,
                          seed: int = 42) -> np.ndarray:
    """Run a static (zero-motion) simulation for Allan variance analysis.

    Args:
        params: ImuNbkParams from YAML config.
        duration_s: Duration in seconds.
        seed: RNG seed for reproducibility.

    Returns:
        Array of shape (n_samples, 6) with columns:
            [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    """
    n_samples = int(duration_s / params.dt)
    rng = np.random.default_rng(seed)
    state = init_state()
    output = np.zeros((n_samples, 6))

    for i in range(n_samples):
        state, noise = step(state, params, rng)
        output[i] = noise

    return output
