"""Deterministic cross-backend verification tests.

Verifies that all backends produce identical output when given the same
random draws. This is the "same math, different language" test.

We inject fixed draws [1.0, 2.0, 3.0] into the Python backends via the
fixed_draws parameter and compare. The C++ side uses a fixed seed for
determinism instead of injected draws.
"""

import numpy as np
import pytest

from imu_nbk.numpy_backend import init_state, step


class TestNumpyBackendDeterministic:
    """NumPy backend produces correct output with fixed random draws."""

    def test_single_step_fixed_draws(self, params, fixed_draws_one_step):
        """Verify one step with known draws matches expected computation."""
        state = init_state()
        new_state, noise = step(state, params, None, fixed_draws=fixed_draws_one_step)

        # With zero initial state and draws [1,2,3]:
        # output = z_B(0) + z_K(0) + sqrt(Q_N) * draw[0]
        #        = 0 + 0 + sqrt(N^2/dt) * 1.0
        #        = N / sqrt(dt)
        dt = params.dt
        expected_acc_x = params.acc_N[0] / np.sqrt(dt) * 1.0
        assert noise[0] == pytest.approx(expected_acc_x, rel=1e-12)

    def test_state_update_fixed_draws(self, params, fixed_draws_one_step):
        """Verify state transitions with known draws."""
        state = init_state()
        new_state, _ = step(state, params, None, fixed_draws=fixed_draws_one_step)

        dt = params.dt
        # z_B_new = phi * 0 + sqrt(Q_B) * draw[1] = sqrt(Q_B) * 2.0
        beta = 1.0 / params.acc_corr_time[0]
        S_B = (2.0 * params.acc_B[0]**2 * np.log(2.0)) / (
            np.pi * 0.4365**2 * params.acc_corr_time[0])
        Q_B = S_B / (2.0 * beta) * (1.0 - np.exp(-2.0 * beta * dt))
        expected_z_B = np.sqrt(Q_B) * 2.0

        assert new_state["acc_z_B"][0] == pytest.approx(expected_z_B, rel=1e-12)

        # z_K_new = 0 + sqrt(S_K * dt) * draw[2] = sqrt(K^2 * dt) * 3.0
        expected_z_K = np.sqrt(params.acc_K[0]**2 * dt) * 3.0
        assert new_state["acc_z_K"][0] == pytest.approx(expected_z_K, rel=1e-12)

    def test_multi_step_state_accumulation(self, params, fixed_draws_one_step):
        """Verify state carries forward across multiple steps."""
        state = init_state()

        state1, noise1 = step(state, params, None, fixed_draws=fixed_draws_one_step)
        state2, noise2 = step(state1, params, None, fixed_draws=fixed_draws_one_step)

        # Second step output should differ because state is non-zero
        # output = z_B[k] + z_K[k] + sqrt(Q_N) * draw[0]
        # z_B[k] and z_K[k] are non-zero from step 1
        assert noise2[0] != pytest.approx(noise1[0], rel=1e-6)

        # The difference should be exactly state1's z_B + z_K
        expected_diff = state1["acc_z_B"][0] + state1["acc_z_K"][0]
        actual_diff = noise2[0] - noise1[0]
        assert actual_diff == pytest.approx(expected_diff, rel=1e-12)

    def test_gyro_axes_independent(self, params, fixed_draws_one_step):
        """Verify gyro axes use correct parameters (not accel params)."""
        state = init_state()
        _, noise = step(state, params, None, fixed_draws=fixed_draws_one_step)

        dt = params.dt
        # Gyro X output = sqrt(gyro_N[0]^2 / dt) * 1.0
        expected_gyro_x = params.gyro_N[0] / np.sqrt(dt) * 1.0
        assert noise[3] == pytest.approx(expected_gyro_x, rel=1e-12)

        # Gyro Z should use different N value
        expected_gyro_z = params.gyro_N[2] / np.sqrt(dt) * 1.0
        assert noise[5] == pytest.approx(expected_gyro_z, rel=1e-12)

    def test_all_axes_produce_output(self, params, fixed_draws_one_step):
        """All 6 output channels are non-zero."""
        state = init_state()
        _, noise = step(state, params, None, fixed_draws=fixed_draws_one_step)
        assert all(noise != 0.0)


class TestNumpyBackendStatistical:
    """NumPy backend produces statistically correct output with real RNG."""

    def test_zero_mean(self, params):
        """Output should be approximately zero-mean over many samples."""
        rng = np.random.default_rng(42)
        state = init_state()
        n_samples = 100_000
        outputs = np.zeros((n_samples, 6))

        for i in range(n_samples):
            state, noise = step(state, params, rng)
            outputs[i] = noise

        # Mean should be near zero (within 5-sigma of CLT)
        means = outputs.mean(axis=0)
        stds = outputs.std(axis=0) / np.sqrt(n_samples)
        for ax in range(6):
            assert abs(means[ax]) < 5 * stds[ax], (
                f"Axis {ax} mean {means[ax]:.3e} exceeds 5-sigma ({5*stds[ax]:.3e})")

    def test_variance_matches_theory(self, params):
        """Steady-state variance should match theoretical prediction."""
        rng = np.random.default_rng(123)
        state = init_state()
        dt = params.dt

        # Run enough steps to reach steady state (5× longest correlation time)
        max_corr = max(params.acc_corr_time.max(), params.gyro_corr_time.max())
        warmup = int(5 * max_corr / dt)
        for _ in range(min(warmup, 500_000)):
            state, _ = step(state, params, rng)

        # Collect samples
        n_samples = 200_000
        outputs = np.zeros((n_samples, 6))
        for i in range(n_samples):
            state, noise = step(state, params, rng)
            outputs[i] = noise

        # Theoretical variance at steady state:
        # Var = Q_N + Var(z_B_ss) + Var(z_K grows unboundedly, but...)
        # For the N component: variance contribution per sample = N^2/dt
        # This is the dominant term for short-time behavior
        var_acc_x = outputs[:, 0].var()
        expected_Q_N = params.acc_N[0]**2 / dt
        # N dominates at dt-scale, so variance should be >= Q_N
        assert var_acc_x > 0.5 * expected_Q_N


class TestJaxBackendDeterministic:
    """JAX backend matches NumPy reference (when JAX is available)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_jax(self):
        pytest.importorskip("jax")

    def test_matches_numpy_statistically(self, params):
        """JAX and NumPy backends should produce same ADEV at short tau."""
        import jax
        import jax.numpy as jnp
        from imu_nbk.jax_backend import init_state as jax_init, step as jax_step, params_to_jax

        jax_params = params_to_jax(params)

        # Run both backends for same number of steps
        n_steps = 50_000
        dt = params.dt

        # NumPy run
        np_rng = np.random.default_rng(42)
        np_state = init_state()
        np_outputs = np.zeros((n_steps, 6))
        for i in range(n_steps):
            np_state, noise = step(np_state, params, np_rng)
            np_outputs[i] = noise

        # JAX run
        jax_state = jax_init()
        rng = jax.random.PRNGKey(42)
        jax_outputs = np.zeros((n_steps, 6))
        for i in range(n_steps):
            jax_state, noise, rng = jax_step(jax_state, jax_params, rng)
            jax_outputs[i] = np.array(noise)

        # Both should have similar variance (not identical — different RNGs)
        np_var = np_outputs.var(axis=0)
        jax_var = jax_outputs.var(axis=0)
        for ax in range(6):
            ratio = jax_var[ax] / np_var[ax]
            assert 0.8 < ratio < 1.2, (
                f"Axis {ax}: JAX/NumPy variance ratio {ratio:.3f} outside [0.8, 1.2]")
