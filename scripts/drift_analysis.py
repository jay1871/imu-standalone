#!/usr/bin/env python3
"""Long-duration IMU drift/bias visualization.

Runs the NBK stochastic error model for hours with a stationary IMU (zero true
input) and shows how errors accumulate over time:
  - Raw noise per axis (N + B + K contributions)
  - Gyro integrated → attitude drift (degrees)
  - Accel integrated → velocity drift (m/s) and position drift (m)
  - Internal Gauss-Markov state (z_B) and random walk state (z_K)

No MuJoCo needed — this is pure error model analysis.

Usage:
  python drift_analysis.py                         # 10 hours, JAX if available
  python drift_analysis.py --duration 5            # 5 hours
  python drift_analysis.py --backend numpy         # force NumPy (slower)
  python drift_analysis.py --save drift.csv --no-plot  # CSV only, headless
  python drift_analysis.py --rate 100              # 100 Hz (faster, less data)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from imu_nbk import load_config


def run_numpy(params, n_steps, dt, decimate):
    """Run simulation with NumPy backend, decimating on the fly."""
    from imu_nbk.numpy_backend import _nbk_axis_update

    n_output = n_steps // decimate
    rng = np.random.default_rng(42)

    # Storage (decimated)
    times = np.zeros(n_output)
    noise_out = np.zeros((n_output, 6))
    z_B_out = np.zeros((n_output, 6))
    z_K_out = np.zeros((n_output, 6))
    attitude_out = np.zeros((n_output, 3))
    velocity_out = np.zeros((n_output, 3))
    position_out = np.zeros((n_output, 3))

    # State
    acc_z_B = np.zeros(3)
    acc_z_K = np.zeros(3)
    gyro_z_B = np.zeros(3)
    gyro_z_K = np.zeros(3)

    # Running integrals
    attitude = np.zeros(3)   # integrated gyro noise (rad)
    velocity = np.zeros(3)   # integrated accel noise (m/s)
    position = np.zeros(3)   # double-integrated accel noise (m)

    out_idx = 0
    wall_start = time.perf_counter()
    last_print = wall_start

    for i in range(n_steps):
        # Accelerometer
        noise = np.zeros(6)
        for ax in range(3):
            acc_z_B[ax], acc_z_K[ax], noise[ax] = _nbk_axis_update(
                acc_z_B[ax], acc_z_K[ax],
                params.acc_N[ax], params.acc_B[ax], params.acc_K[ax],
                params.acc_corr_time[ax], dt, rng)

        # Gyroscope
        for ax in range(3):
            gyro_z_B[ax], gyro_z_K[ax], noise[3 + ax] = _nbk_axis_update(
                gyro_z_B[ax], gyro_z_K[ax],
                params.gyro_N[ax], params.gyro_B[ax], params.gyro_K[ax],
                params.gyro_corr_time[ax], dt, rng)

        # Integrate at full rate
        attitude += noise[3:6] * dt
        velocity += noise[0:3] * dt
        position += velocity * dt

        # Store decimated
        if (i + 1) % decimate == 0:
            times[out_idx] = (i + 1) * dt
            noise_out[out_idx] = noise
            z_B_out[out_idx, :3] = acc_z_B
            z_B_out[out_idx, 3:] = gyro_z_B
            z_K_out[out_idx, :3] = acc_z_K
            z_K_out[out_idx, 3:] = gyro_z_K
            attitude_out[out_idx] = attitude
            velocity_out[out_idx] = velocity
            position_out[out_idx] = position
            out_idx += 1

        # Print progress every 5 seconds
        now = time.perf_counter()
        if now - last_print >= 5.0:
            last_print = now
            pct = 100.0 * (i + 1) / n_steps
            elapsed = now - wall_start
            if pct > 0:
                eta = elapsed * (100.0 / pct - 1.0)
                eta_min = int(eta) // 60
                eta_sec = int(eta) % 60
                eta_str = f"{eta_min}m {eta_sec:02d}s"
            else:
                eta_str = "calculating..."
            hrs = (i + 1) * dt / 3600
            print(f"  {pct:5.1f}% ({hrs:.2f} hrs) | ETA: {eta_str} | "
                  f"attitude drift: [{attitude[0]*180/np.pi:+.3f}, "
                  f"{attitude[1]*180/np.pi:+.3f}, {attitude[2]*180/np.pi:+.3f}] deg")

    return {
        "times": times[:out_idx],
        "noise": noise_out[:out_idx],
        "z_B": z_B_out[:out_idx],
        "z_K": z_K_out[:out_idx],
        "attitude": attitude_out[:out_idx],
        "velocity": velocity_out[:out_idx],
        "position": position_out[:out_idx],
    }


def run_jax(params, n_steps, dt, decimate):
    """Run simulation with JAX backend using lax.scan (much faster)."""
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

    import jax
    import jax.numpy as jnp
    from jax import random

    print(f"  JAX backend: {jax.default_backend()}")

    # Parameters as JAX arrays
    acc_N = jnp.array(params.acc_N)
    acc_B = jnp.array(params.acc_B)
    acc_K = jnp.array(params.acc_K)
    acc_corr = jnp.array(params.acc_corr_time)
    gyro_N = jnp.array(params.gyro_N)
    gyro_B = jnp.array(params.gyro_B)
    gyro_K = jnp.array(params.gyro_K)
    gyro_corr = jnp.array(params.gyro_corr_time)

    # Precompute constants
    acc_S_N = acc_N ** 2
    acc_S_B = (2.0 * acc_B**2 * jnp.log(2.0)) / (jnp.pi * 0.4365**2 * acc_corr)
    acc_S_K = acc_K ** 2
    acc_beta = 1.0 / acc_corr
    acc_phi = jnp.exp(-acc_beta * dt)
    acc_Q_B = acc_S_B / (2.0 * acc_beta) * (1.0 - jnp.exp(-2.0 * acc_beta * dt))
    acc_Q_K = acc_S_K * dt
    acc_Q_N = acc_S_N / dt

    gyro_S_N = gyro_N ** 2
    gyro_S_B = (2.0 * gyro_B**2 * jnp.log(2.0)) / (jnp.pi * 0.4365**2 * gyro_corr)
    gyro_S_K = gyro_K ** 2
    gyro_beta = 1.0 / gyro_corr
    gyro_phi = jnp.exp(-gyro_beta * dt)
    gyro_Q_B = gyro_S_B / (2.0 * gyro_beta) * (1.0 - jnp.exp(-2.0 * gyro_beta * dt))
    gyro_Q_K = gyro_S_K * dt
    gyro_Q_N = gyro_S_N / dt

    n_output = n_steps // decimate

    # Single-step function (no inner scan output — avoids storing intermediates)
    @jax.jit
    def jax_step(carry, _):
        az_B, az_K, gz_B, gz_K, att, vel, pos, key = carry

        # Split into 18 sub-keys (3 draws × 6 axes) + 1 carry
        keys = random.split(key, 19)
        key = keys[0]

        # Accelerometer (3 axes)
        acc_noise = jnp.zeros(3)
        new_az_B = jnp.zeros(3)
        new_az_K = jnp.zeros(3)
        for ax in range(3):
            k1, k2, k3 = random.split(keys[1 + ax], 3)
            w_N = jnp.sqrt(acc_Q_N[ax]) * random.normal(k1)
            w_B = jnp.sqrt(acc_Q_B[ax]) * random.normal(k2)
            w_K = jnp.sqrt(acc_Q_K[ax]) * random.normal(k3)
            acc_noise = acc_noise.at[ax].set(az_B[ax] + az_K[ax] + w_N)
            new_az_B = new_az_B.at[ax].set(acc_phi[ax] * az_B[ax] + w_B)
            new_az_K = new_az_K.at[ax].set(az_K[ax] + w_K)

        # Gyroscope (3 axes)
        gyro_noise = jnp.zeros(3)
        new_gz_B = jnp.zeros(3)
        new_gz_K = jnp.zeros(3)
        for ax in range(3):
            k1, k2, k3 = random.split(keys[4 + ax], 3)
            w_N = jnp.sqrt(gyro_Q_N[ax]) * random.normal(k1)
            w_B = jnp.sqrt(gyro_Q_B[ax]) * random.normal(k2)
            w_K = jnp.sqrt(gyro_Q_K[ax]) * random.normal(k3)
            gyro_noise = gyro_noise.at[ax].set(gz_B[ax] + gz_K[ax] + w_N)
            new_gz_B = new_gz_B.at[ax].set(gyro_phi[ax] * gz_B[ax] + w_B)
            new_gz_K = new_gz_K.at[ax].set(gz_K[ax] + w_K)

        # Integrate
        att = att + gyro_noise * dt
        vel = vel + acc_noise * dt
        pos = pos + vel * dt

        new_carry = (new_az_B, new_az_K, new_gz_B, new_gz_K, att, vel, pos, key)
        return new_carry, None

    # Decimated step: run `decimate` inner steps, output final state
    @jax.jit
    def decimated_step(carry, _):
        carry, _ = jax.lax.scan(jax_step, carry, None, length=decimate)
        az_B, az_K, gz_B, gz_K, att, vel, pos, key = carry
        noise_sample = jnp.concatenate([az_B + az_K, gz_B + gz_K])
        z_B_sample = jnp.concatenate([az_B, gz_B])
        z_K_sample = jnp.concatenate([az_K, gz_K])
        output = (noise_sample, z_B_sample, z_K_sample, att, vel, pos)
        return carry, output

    # Initial state
    init_carry = (
        jnp.zeros(3), jnp.zeros(3),  # acc z_B, z_K
        jnp.zeros(3), jnp.zeros(3),  # gyro z_B, z_K
        jnp.zeros(3), jnp.zeros(3), jnp.zeros(3),  # attitude, velocity, position
        random.PRNGKey(42),
    )

    # Run in small batches to control memory and show progress.
    # Each batch = 100 decimated outputs = 100 * decimate steps.
    BATCH_SIZE = 100
    n_batches = (n_output + BATCH_SIZE - 1) // BATCH_SIZE

    @jax.jit
    def run_batch(carry):
        return jax.lax.scan(decimated_step, carry, None, length=BATCH_SIZE)

    print(f"  Running {n_output} decimated steps in {n_batches} batches of {BATCH_SIZE}...")
    print(f"  ({BATCH_SIZE * decimate} raw steps per batch)")
    print(f"  JIT compiling first batch (may take 30-60s)...")

    all_outputs = []
    carry = init_carry
    t0 = time.perf_counter()
    last_print = t0
    completed = 0

    for batch_i in range(n_batches):
        carry, batch_out = run_batch(carry)
        batch_out[0].block_until_ready()

        all_outputs.append(batch_out)
        completed += BATCH_SIZE

        now = time.perf_counter()
        if now - last_print >= 5.0 or batch_i == n_batches - 1:
            last_print = now
            pct = min(100.0, 100.0 * completed / n_output)
            elapsed = now - t0
            if pct > 0 and pct < 100.0:
                eta = elapsed * (100.0 / pct - 1.0)
                eta_min = int(eta) // 60
                eta_sec = int(eta) % 60
                eta_str = f"{eta_min}m {eta_sec:02d}s"
            else:
                eta_str = "done"
            print(f"  {pct:5.1f}% | ETA: {eta_str} | "
                  f"elapsed: {int(elapsed)//60}m {int(elapsed)%60:02d}s")

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s ({n_steps / elapsed / 1e6:.1f}M steps/s)")

    # Concatenate and trim to actual output length
    outputs = jax.tree.map(lambda *arrs: jnp.concatenate(arrs, axis=0)[:n_output], *all_outputs)

    noise_out, z_B_out, z_K_out, att_out, vel_out, pos_out = outputs
    times = np.arange(1, n_output + 1) * (decimate * dt)

    return {
        "times": times,
        "noise": np.array(noise_out),
        "z_B": np.array(z_B_out),
        "z_K": np.array(z_K_out),
        "attitude": np.array(att_out),
        "velocity": np.array(vel_out),
        "position": np.array(pos_out),
    }


def plot_drift(results, params, duration_hrs):
    """Generate the multi-panel drift visualization."""
    import matplotlib.pyplot as plt

    t_hrs = results["times"] / 3600.0
    axis_labels = ["X", "Y", "Z"]
    colors = ["tab:red", "tab:green", "tab:blue"]

    fig = plt.figure(figsize=(16, 20))
    fig.suptitle(f"IMU Drift Analysis — {duration_hrs:.0f} Hours Stationary (ADIS16607)",
                 fontsize=14, y=0.98)

    # --- Row 1: Raw noise ---
    for i in range(3):
        ax = fig.add_subplot(5, 3, i + 1)
        ax.plot(t_hrs, results["noise"][:, i], color=colors[i], linewidth=0.3, alpha=0.7)
        ax.set_ylabel(f"Accel {axis_labels[i]} (m/s²)")
        ax.set_title(f"Accelerometer {axis_labels[i]}")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    for i in range(3):
        ax = fig.add_subplot(5, 3, i + 4)
        ax.plot(t_hrs, results["noise"][:, 3 + i] * 180.0 / np.pi, color=colors[i], linewidth=0.3, alpha=0.7)
        ax.set_ylabel(f"Gyro {axis_labels[i]} (deg/s)")
        ax.set_title(f"Gyroscope {axis_labels[i]}")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    # --- Row 2: Attitude drift (integrated gyro) ---
    att_deg = results["attitude"] * 180.0 / np.pi
    for i in range(3):
        ax = fig.add_subplot(5, 3, i + 7)
        ax.plot(t_hrs, att_deg[:, i], color=colors[i], linewidth=0.8)
        ax.set_ylabel(f"Attitude {axis_labels[i]} (deg)")
        ax.set_title(f"Attitude Drift {axis_labels[i]}")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    # --- Row 3: Position drift (double-integrated accel) ---
    for i in range(3):
        ax = fig.add_subplot(5, 3, i + 10)
        ax.plot(t_hrs, results["position"][:, i], color=colors[i], linewidth=0.8)
        ax.set_ylabel(f"Position {axis_labels[i]} (m)")
        ax.set_title(f"Position Drift {axis_labels[i]}")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    # --- Row 4: Internal states z_B and z_K ---
    for i in range(3):
        ax = fig.add_subplot(5, 3, i + 13)
        ax.plot(t_hrs, results["z_B"][:, 3 + i] * 180.0 / np.pi, color=colors[i], linewidth=0.5,
                alpha=0.8, label="z_B (Gauss-Markov)")
        ax.plot(t_hrs, results["z_K"][:, 3 + i] * 180.0 / np.pi, color=colors[i], linewidth=0.8,
                alpha=0.6, linestyle="--", label="z_K (random walk)")
        ax.set_ylabel(f"Gyro {axis_labels[i]} state (deg/s)")
        ax.set_xlabel("Time (hours)")
        ax.set_title(f"Internal States — Gyro {axis_labels[i]}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("drift_analysis.png", dpi=150, bbox_inches="tight")
    print(f"Saved: drift_analysis.png")
    plt.show()


def plot_drift_summary(results, params, duration_hrs):
    """Single-page summary: attitude + position drift with envelope."""
    import matplotlib.pyplot as plt

    t_hrs = results["times"] / 3600.0
    t_s = results["times"]
    axis_labels = ["X", "Y", "Z"]
    colors = ["tab:red", "tab:green", "tab:blue"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"IMU Navigation Drift — {duration_hrs:.0f} Hours Stationary (ADIS16607)",
                 fontsize=13)

    # Attitude drift
    att_deg = results["attitude"] * 180.0 / np.pi
    for i in range(3):
        axes[0].plot(t_hrs, att_deg[:, i], color=colors[i], linewidth=0.8,
                     label=f"{axis_labels[i]} ({att_deg[-1, i]:+.2f} deg final)")
    axes[0].set_ylabel("Attitude Drift (degrees)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="black", linewidth=0.5, alpha=0.3)
    axes[0].set_title("Gyroscope Integration → Attitude Error")

    # Position drift
    for i in range(3):
        axes[1].plot(t_hrs, results["position"][:, i], color=colors[i], linewidth=0.8,
                     label=f"{axis_labels[i]} ({results['position'][-1, i]:+.1f} m final)")
    axes[1].set_ylabel("Position Drift (meters)")
    axes[1].set_xlabel("Time (hours)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="black", linewidth=0.5, alpha=0.3)
    axes[1].set_title("Accelerometer Double Integration → Position Error")

    plt.tight_layout()
    plt.savefig("drift_summary.png", dpi=150, bbox_inches="tight")
    print(f"Saved: drift_summary.png")
    plt.show()


def save_csv(results, output_path):
    """Save decimated results to CSV."""
    n = len(results["times"])
    data = np.column_stack([
        results["times"],
        results["noise"],
        results["z_B"],
        results["z_K"],
        results["attitude"],
        results["velocity"],
        results["position"],
    ])
    header = (
        "time,"
        "noise_acc_x,noise_acc_y,noise_acc_z,noise_gyro_x,noise_gyro_y,noise_gyro_z,"
        "zB_acc_x,zB_acc_y,zB_acc_z,zB_gyro_x,zB_gyro_y,zB_gyro_z,"
        "zK_acc_x,zK_acc_y,zK_acc_z,zK_gyro_x,zK_gyro_y,zK_gyro_z,"
        "attitude_x,attitude_y,attitude_z,"
        "velocity_x,velocity_y,velocity_z,"
        "position_x,position_y,position_z"
    )
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")
    print(f"Saved {n} decimated samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Long-duration IMU drift/bias analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python drift_analysis.py                             # 10 hours, auto backend
  python drift_analysis.py --duration 5                # 5 hours
  python drift_analysis.py --duration 1 --backend numpy  # 1 hour, NumPy only
  python drift_analysis.py --rate 100 --duration 10    # 100 Hz (10x faster)
  python drift_analysis.py --save drift_10hr.csv --no-plot
""")

    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration in hours (default: 10)")
    parser.add_argument("--rate", type=float, default=None,
                        help="Sample rate Hz (default: from YAML config, usually 1000)")
    parser.add_argument("--decimate", type=int, default=100,
                        help="Decimation factor for output (default: 100 → 10 Hz output)")
    parser.add_argument("--backend", choices=["auto", "jax", "numpy"], default="auto",
                        help="Computation backend (default: auto = JAX if available)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save decimated CSV to this path")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting (just compute and save CSV)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only show the 2-panel summary plot (attitude + position)")

    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).resolve().parent.parent / "config" / "adis16607.yaml"
    params = load_config(str(config_path))

    # Override rate if requested
    if args.rate is not None:
        params = params._replace(dt=1.0 / args.rate) if hasattr(params, '_replace') else params
        # params is a dataclass, so override dt
        object.__setattr__(params, 'dt', 1.0 / args.rate)

    dt = params.dt
    duration_s = args.duration * 3600.0
    n_steps = int(duration_s / dt)
    decimate = args.decimate

    print(f"IMU Drift Analysis")
    print(f"  Duration: {args.duration} hours ({duration_s:.0f} s)")
    print(f"  Sample rate: {1.0/dt:.0f} Hz")
    print(f"  Total steps: {n_steps:,}")
    print(f"  Decimation: {decimate}x → {n_steps // decimate:,} output samples")
    print(f"  Output rate: {1.0/(dt*decimate):.1f} Hz")
    print()

    # Select backend
    use_jax = False
    if args.backend == "jax":
        use_jax = True
    elif args.backend == "auto":
        try:
            import jax
            use_jax = True
        except ImportError:
            pass

    # Run
    t0 = time.perf_counter()
    if use_jax:
        print(f"Backend: JAX")
        results = run_jax(params, n_steps, dt, decimate)
    else:
        print(f"Backend: NumPy (this will be slow for long durations)")
        results = run_numpy(params, n_steps, dt, decimate)
    elapsed = time.perf_counter() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"\nFinal drift after {args.duration:.1f} hours:")
    att_deg = results["attitude"][-1] * 180.0 / np.pi
    print(f"  Attitude:  X={att_deg[0]:+.4f} deg, Y={att_deg[1]:+.4f} deg, Z={att_deg[2]:+.4f} deg")
    print(f"  Velocity:  X={results['velocity'][-1, 0]:+.4f} m/s, "
          f"Y={results['velocity'][-1, 1]:+.4f} m/s, Z={results['velocity'][-1, 2]:+.4f} m/s")
    print(f"  Position:  X={results['position'][-1, 0]:+.2f} m, "
          f"Y={results['position'][-1, 1]:+.2f} m, Z={results['position'][-1, 2]:+.2f} m")

    # Save CSV
    if args.save:
        save_csv(results, args.save)
    else:
        default_csv = f"drift_{args.duration:.0f}hr.csv"
        save_csv(results, default_csv)

    # Plot
    if not args.no_plot:
        try:
            if args.summary_only:
                plot_drift_summary(results, params, args.duration)
            else:
                plot_drift(results, params, args.duration)
                plot_drift_summary(results, params, args.duration)
        except ImportError:
            print("matplotlib not available — skipping plots")
        except Exception as e:
            print(f"Plotting failed ({e}) — CSV saved successfully")


if __name__ == "__main__":
    main()
