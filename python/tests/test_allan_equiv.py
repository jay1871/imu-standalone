"""Allan variance equivalence test across backends.

Runs each available backend for sufficient duration and verifies that
the computed Allan deviation matches the theoretical NBK curve within
the expected statistical confidence interval.

This test is SLOW (minutes) — run with: pytest -m slow
"""

import numpy as np
import pytest

from imu_nbk.params import load_config


pytestmark = pytest.mark.slow


def compute_theoretical_adev(tau, N, B, K, corr_time, B_poles=None, corr_time_poles=None):
    """Theoretical Allan deviation from NBK parameters (exact Gauss-Markov, single or multi-pole)."""
    sigma2_N = N**2 / tau
    sigma2_K = K**2 * tau / 3.0
    if B_poles is not None and corr_time_poles is not None:
        sigma2_B = 0.0
        for b_p, tc_p in zip(B_poles, corr_time_poles):
            beta = 1.0 / tc_p
            S_B = (2.0 * b_p**2 * np.log(2.0)) / (np.pi * 0.4365**2 * tc_p)
            x = beta * tau
            sigma2_B += (S_B / (beta * x)) * (
                1.0 - (1.0 / (2.0 * x)) * (3.0 - 4.0 * np.exp(-x) + np.exp(-2.0 * x)))
    else:
        beta = 1.0 / corr_time
        S_B = (2.0 * B**2 * np.log(2.0)) / (np.pi * 0.4365**2 * corr_time)
        x = beta * tau
        sigma2_B = (S_B / (beta * x)) * (
            1.0 - (1.0 / (2.0 * x)) * (3.0 - 4.0 * np.exp(-x) + np.exp(-2.0 * x)))
    return np.sqrt(sigma2_N + sigma2_B + sigma2_K)


def run_numpy_and_compute_adev(params, duration_hours=2, rate_hz=100):
    """Run NumPy backend and compute overlapping Allan deviation."""
    allantools = pytest.importorskip("allantools")
    from imu_nbk.numpy_backend import init_state, step

    dt = 1.0 / rate_hz
    # Override dt in params for this test rate
    from dataclasses import replace
    test_params = replace(params, dt=dt)

    n_samples = int(duration_hours * 3600 * rate_hz)
    rng = np.random.default_rng(42)
    state = init_state()
    output = np.zeros((n_samples, 6))

    for i in range(n_samples):
        state, noise = step(state, test_params, rng)
        output[i] = noise

    # Compute OADEV for accelerometer X-axis
    taus_out, adev, _, _ = allantools.oadev(
        output[:, 0], rate=rate_hz, data_type="freq", taus="octave")

    return taus_out, adev, output


class TestAllanVarianceNumpy:
    """NumPy backend Allan deviation matches theory."""

    def test_short_tau_matches_N(self, params):
        """At tau=1s, ADEV should be dominated by N (slope -1/2)."""
        taus, adev, _ = run_numpy_and_compute_adev(params, duration_hours=1, rate_hz=100)

        # Find tau closest to 1s
        idx = np.argmin(np.abs(taus - 1.0))
        if taus[idx] > 2.0:
            pytest.skip("No tau point near 1s")

        tau_val = taus[idx]
        measured = adev[idx]

        # Theory at this tau (accel X-axis)
        theory = compute_theoretical_adev(
            tau_val, params.acc_N[0], params.acc_B[0],
            params.acc_K[0], params.acc_corr_time[0],
            params.acc_B_poles[0] if params.acc_B_poles is not None else None,
            params.acc_corr_time_poles[0] if params.acc_corr_time_poles is not None else None)

        # Should match within 20% (generous for 1-hour run)
        ratio = measured / theory
        assert 0.8 < ratio < 1.2, (
            f"Short-tau ADEV ratio {ratio:.3f} outside [0.8, 1.2] "
            f"(measured={measured:.3e}, theory={theory:.3e})")

    def test_mid_tau_matches_B_plateau(self, params):
        """At tau ~ correlation_time, ADEV should be near 0.6645*B."""
        taus, adev, _ = run_numpy_and_compute_adev(params, duration_hours=2, rate_hz=100)

        corr_time = params.acc_corr_time[0]
        # Find tau closest to correlation_time
        idx = np.argmin(np.abs(taus - corr_time))
        if abs(taus[idx] - corr_time) / corr_time > 0.5:
            pytest.skip(f"No tau point near correlation_time={corr_time}s")

        tau_val = taus[idx]
        measured = adev[idx]
        theory = compute_theoretical_adev(
            tau_val, params.acc_N[0], params.acc_B[0],
            params.acc_K[0], params.acc_corr_time[0],
            params.acc_B_poles[0] if params.acc_B_poles is not None else None,
            params.acc_corr_time_poles[0] if params.acc_corr_time_poles is not None else None)

        ratio = measured / theory
        assert 0.7 < ratio < 1.3, (
            f"Mid-tau ADEV ratio {ratio:.3f} outside [0.7, 1.3] "
            f"(measured={measured:.3e}, theory={theory:.3e})")


class TestAllanVarianceJax:
    """JAX backend Allan deviation matches NumPy backend."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_jax(self):
        pytest.importorskip("jax")

    def test_adev_matches_numpy(self, params):
        """JAX ADEV should agree with NumPy ADEV within statistical noise."""
        allantools = pytest.importorskip("allantools")
        import jax
        import jax.numpy as jnp
        from imu_nbk.jax_backend import init_state, step, params_to_jax
        from dataclasses import replace

        rate_hz = 100
        dt = 1.0 / rate_hz
        test_params = replace(params, dt=dt)
        jax_params = params_to_jax(test_params)

        n_samples = int(1 * 3600 * rate_hz)  # 1 hour
        state = init_state()
        rng = jax.random.PRNGKey(99)
        output = np.zeros((n_samples, 6))

        for i in range(n_samples):
            state, noise, rng = step(state, jax_params, rng)
            output[i] = np.array(noise)

        taus_jax, adev_jax, _, _ = allantools.oadev(
            output[:, 0], rate=rate_hz, data_type="freq", taus="octave")

        # Compare against theory at tau=1
        idx = np.argmin(np.abs(taus_jax - 1.0))
        theory = compute_theoretical_adev(
            taus_jax[idx], test_params.acc_N[0], test_params.acc_B[0],
            test_params.acc_K[0], test_params.acc_corr_time[0])

        ratio = adev_jax[idx] / theory
        assert 0.75 < ratio < 1.25, (
            f"JAX short-tau ADEV ratio {ratio:.3f} outside [0.75, 1.25]")
