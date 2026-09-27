#!/usr/bin/env python3
"""Unified IMU simulation script — runs via MuJoCo C plugin OR MJX+JAX backend.

Usage:
  # Mode 1: MuJoCo C plugin (CPU, single environment)
  python simulate_imu.py --mode plugin --model ../models/free_body_imu.xml

  # Mode 2: MJX + JAX backend (GPU, batched environments)
  python simulate_imu.py --mode mjx --model ../models/free_body_imu.xml --n-envs 4

  # Mode 3: JAX backend only, no physics (static/zero-input IMU noise)
  python simulate_imu.py --mode jax-only --duration 60

Both modes produce time-series CSV output and optional plots comparing
noisy vs true sensor readings.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add the python package to path so imu_nbk is importable without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))


def run_plugin_mode(args):
    """Run simulation using the MuJoCo C plugin (CPU, single env).

    The plugin does everything: MuJoCo steps physics, the plugin reads
    true acceleration/angular velocity from mjData, applies the full NBK
    error model, and writes noisy output to sensordata.
    """
    import mujoco

    # Load the plugin shared library
    plugin_path = Path(__file__).resolve().parent.parent / "build" / "mujoco_plugin" / "imu_nbk_plugin.so"
    if not plugin_path.exists():
        sys.exit(f"Plugin not found at {plugin_path}\n"
                 f"Build with: cmake .. -DIMU_BUILD_MUJOCO_PLUGIN=ON -DMUJOCO_ROOT=...")
    mujoco.mj_loadPluginLibrary(str(plugin_path))

    # Load model
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)

    n_steps = int(args.duration / m.opt.timestep)
    dt = m.opt.timestep

    # Find sensor addresses
    acc_true_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "acc_true")]
    gyro_true_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro_true")]
    imu_noisy_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_noisy")]

    # Storage
    times = np.zeros(n_steps)
    acc_true = np.zeros((n_steps, 3))
    gyro_true = np.zeros((n_steps, 3))
    acc_noisy = np.zeros((n_steps, 3))
    gyro_noisy = np.zeros((n_steps, 3))

    # Optional: apply control (for spinning_body model)
    has_motor = m.nu > 0

    print(f"Running MuJoCo C plugin simulation: {n_steps} steps, dt={dt*1000:.1f}ms")
    mujoco.mj_resetData(m, d)

    for i in range(n_steps):
        if has_motor and args.ctrl is not None:
            d.ctrl[0] = args.ctrl

        mujoco.mj_step(m, d)

        times[i] = d.time
        acc_true[i] = d.sensordata[acc_true_adr:acc_true_adr + 3]
        gyro_true[i] = d.sensordata[gyro_true_adr:gyro_true_adr + 3]
        acc_noisy[i] = d.sensordata[imu_noisy_adr:imu_noisy_adr + 3]
        gyro_noisy[i] = d.sensordata[imu_noisy_adr + 3:imu_noisy_adr + 6]

    results = {
        "times": times,
        "acc_true": acc_true,
        "gyro_true": gyro_true,
        "acc_noisy": acc_noisy,
        "gyro_noisy": gyro_noisy,
    }
    return results, dt


def run_mjx_mode(args):
    """Run simulation using MJX (JAX) for physics + imu_nbk JAX backend for noise.

    MJX steps the physics on GPU. It provides noiseless sensor readings via
    built-in accelerometer/gyro sensors. We then pass those through the JAX
    NBK error model to get realistic noisy IMU output.

    This is the pattern you'd use in batched RL training.
    """
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    from imu_nbk import load_config
    from imu_nbk.jax_backend import init_state, init_state_batched, step, params_to_jax, batched_step

    # Load MuJoCo model (CPU) then put on device
    model_path = Path(args.model).resolve()
    m = mujoco.MjModel.from_xml_path(str(model_path))
    mx = mjx.put_model(m)

    dt = m.opt.timestep
    n_steps = int(args.duration / dt)
    n_envs = args.n_envs

    # Find noiseless sensor addresses
    acc_true_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "acc_true")]
    gyro_true_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro_true")]

    # Load IMU error model params
    config_path = Path(__file__).resolve().parent.parent / "config" / "adis16607.yaml"
    params = load_config(str(config_path))
    jax_params = params_to_jax(params)

    # Initialize batched MJX data
    d = mujoco.MjData(m)
    dx = mjx.put_data(m, d)
    # Replicate across environments
    dx_batched = jax.vmap(lambda _: dx)(jnp.arange(n_envs))

    # Initialize IMU error model state (one per environment)
    imu_states = init_state_batched(n_envs)
    rng_keys = jax.random.split(jax.random.PRNGKey(0), n_envs)

    # JIT-compile the step functions
    @jax.jit
    def physics_step(dx_batch):
        return jax.vmap(lambda dx: mjx.step(mx, dx))(dx_batch)

    @jax.jit
    def imu_step(imu_states, jax_params, rng_keys):
        return batched_step(imu_states, jax_params, rng_keys)

    # Storage (just env 0 for output)
    times = np.zeros(n_steps)
    acc_true_out = np.zeros((n_steps, 3))
    gyro_true_out = np.zeros((n_steps, 3))
    acc_noisy_out = np.zeros((n_steps, 3))
    gyro_noisy_out = np.zeros((n_steps, 3))

    print(f"Running MJX + JAX simulation: {n_steps} steps, dt={dt*1000:.1f}ms, "
          f"{n_envs} parallel environments")
    print(f"  JAX backend: {jax.default_backend()}")

    for i in range(n_steps):
        # Step physics (all envs in parallel on GPU)
        dx_batched = physics_step(dx_batched)

        # Read noiseless sensor data from MJX
        # sensordata shape: (n_envs, n_sensordata)
        true_acc = dx_batched.sensordata[:, acc_true_adr:acc_true_adr + 3]
        true_gyro = dx_batched.sensordata[:, gyro_true_adr:gyro_true_adr + 3]

        # Apply NBK error model (all envs in parallel)
        imu_states, noises, rng_keys = imu_step(imu_states, jax_params, rng_keys)

        # Noisy = true + stochastic noise
        noisy_acc = true_acc + noises[:, :3]
        noisy_gyro = true_gyro + noises[:, 3:]

        # Store env 0 for output
        times[i] = i * dt
        acc_true_out[i] = np.array(true_acc[0])
        gyro_true_out[i] = np.array(true_gyro[0])
        acc_noisy_out[i] = np.array(noisy_acc[0])
        gyro_noisy_out[i] = np.array(noisy_gyro[0])

    results = {
        "times": times,
        "acc_true": acc_true_out,
        "gyro_true": gyro_true_out,
        "acc_noisy": acc_noisy_out,
        "gyro_noisy": gyro_noisy_out,
    }
    return results, dt


def run_jax_only_mode(args):
    """Run JAX IMU error model without physics (static/zero-input scenario).

    Useful for Allan variance validation or studying noise characteristics
    without needing MuJoCo installed.
    """
    import jax
    import jax.numpy as jnp

    from imu_nbk import load_config
    from imu_nbk.jax_backend import init_state, step, params_to_jax

    config_path = Path(__file__).resolve().parent.parent / "config" / "adis16607.yaml"
    params = load_config(str(config_path))
    jax_params = params_to_jax(params)

    dt = params.dt
    n_steps = int(args.duration / dt)

    state = init_state()
    rng = jax.random.PRNGKey(args.seed)

    # Use jax.lax.scan for efficiency
    @jax.jit
    def scan_fn(carry, _):
        st, key = carry
        st, noise, key = step(st, jax_params, key)
        return (st, key), noise

    print(f"Running JAX-only simulation (no physics): {n_steps} steps, dt={dt*1000:.1f}ms")
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"  Duration: {args.duration}s")

    (_, _), noises = jax.lax.scan(scan_fn, (state, rng), None, length=n_steps)
    noises = np.array(noises)

    times = np.arange(n_steps) * dt
    results = {
        "times": times,
        "acc_true": np.zeros((n_steps, 3)),
        "gyro_true": np.zeros((n_steps, 3)),
        "acc_noisy": noises[:, :3],
        "gyro_noisy": noises[:, 3:],
    }
    return results, dt


def save_csv(results, output_path, dt):
    """Save results to CSV."""
    n = len(results["times"])
    data = np.column_stack([
        results["times"],
        results["acc_true"],
        results["gyro_true"],
        results["acc_noisy"],
        results["gyro_noisy"],
    ])
    header = ("time,"
              "acc_true_x,acc_true_y,acc_true_z,"
              "gyro_true_x,gyro_true_y,gyro_true_z,"
              "acc_noisy_x,acc_noisy_y,acc_noisy_z,"
              "gyro_noisy_x,gyro_noisy_y,gyro_noisy_z")
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")
    print(f"Saved {n} samples to {output_path}")


def plot_results(results, dt, title="IMU Simulation"):
    """Plot true vs noisy sensor data."""
    import matplotlib.pyplot as plt

    t = results["times"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.suptitle(title)

    axis_labels = ["X", "Y", "Z"]

    # Accelerometer
    for ax_i in range(3):
        ax = axes[0, ax_i]
        ax.plot(t, results["acc_true"][:, ax_i], "b-", alpha=0.7, label="True", linewidth=0.5)
        ax.plot(t, results["acc_noisy"][:, ax_i], "r-", alpha=0.5, label="Noisy", linewidth=0.3)
        ax.set_ylabel(f"Accel {axis_labels[ax_i]} (m/s²)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Gyroscope
    for ax_i in range(3):
        ax = axes[1, ax_i]
        ax.plot(t, results["gyro_true"][:, ax_i], "b-", alpha=0.7, label="True", linewidth=0.5)
        ax.plot(t, results["gyro_noisy"][:, ax_i], "r-", alpha=0.5, label="Noisy", linewidth=0.3)
        ax.set_ylabel(f"Gyro {axis_labels[ax_i]} (rad/s)")
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("imu_simulation.png", dpi=150)
    print("Saved plot: imu_simulation.png")
    plt.show()


def plot_noise_comparison(results, dt, title="IMU Noise (Noisy - True)"):
    """Plot just the noise component (difference between noisy and true)."""
    import matplotlib.pyplot as plt

    t = results["times"]
    acc_noise = results["acc_noisy"] - results["acc_true"]
    gyro_noise = results["gyro_noisy"] - results["gyro_true"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title)

    axes[0].plot(t, acc_noise[:, 0], "r-", alpha=0.6, linewidth=0.3, label="X")
    axes[0].plot(t, acc_noise[:, 1], "g-", alpha=0.6, linewidth=0.3, label="Y")
    axes[0].plot(t, acc_noise[:, 2], "b-", alpha=0.6, linewidth=0.3, label="Z")
    axes[0].set_ylabel("Accel noise (m/s²)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, gyro_noise[:, 0], "r-", alpha=0.6, linewidth=0.3, label="X")
    axes[1].plot(t, gyro_noise[:, 1], "g-", alpha=0.6, linewidth=0.3, label="Y")
    axes[1].plot(t, gyro_noise[:, 2], "b-", alpha=0.6, linewidth=0.3, label="Z")
    axes[1].set_ylabel("Gyro noise (rad/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("imu_noise.png", dpi=150)
    print("Saved plot: imu_noise.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Unified IMU simulation — MuJoCo plugin or MJX+JAX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Free-falling body with C plugin (needs plugin .so built)
  python simulate_imu.py --mode plugin --model ../models/free_body_imu.xml --duration 5

  # Spinning body with C plugin
  python simulate_imu.py --mode plugin --model ../models/spinning_body_imu.xml --ctrl 0.5

  # Free body with MJX on GPU (batched)
  python simulate_imu.py --mode mjx --model ../models/free_body_imu.xml --n-envs 16

  # Pure noise (no physics, no mujoco needed — just JAX)
  python simulate_imu.py --mode jax-only --duration 60

  # Save CSV without plotting
  python simulate_imu.py --mode plugin --model ../models/free_body_imu.xml --no-plot -o data.csv
""")

    parser.add_argument("--mode", choices=["plugin", "mjx", "jax-only"], required=True,
                        help="Simulation mode: "
                             "'plugin' = MuJoCo C plugin (CPU), "
                             "'mjx' = MJX physics + JAX noise (GPU), "
                             "'jax-only' = JAX noise only, no physics")
    parser.add_argument("--model", type=str, default="../models/free_body_imu.xml",
                        help="Path to MuJoCo XML model (for plugin/mjx modes)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulation duration in seconds (default: 5)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments (mjx mode, default: 4)")
    parser.add_argument("--ctrl", type=float, default=None,
                        help="Motor control value (for spinning_body model)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (jax-only mode)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV path (default: imu_sim_<mode>.csv)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting (just save CSV)")

    args = parser.parse_args()

    # Run simulation
    if args.mode == "plugin":
        results, dt = run_plugin_mode(args)
    elif args.mode == "mjx":
        results, dt = run_mjx_mode(args)
    elif args.mode == "jax-only":
        results, dt = run_jax_only_mode(args)

    # Save output
    output_path = args.output or f"imu_sim_{args.mode.replace('-', '_')}.csv"
    save_csv(results, output_path, dt)

    # Plot
    if not args.no_plot:
        try:
            model_name = Path(args.model).stem if args.mode != "jax-only" else "static"
            plot_results(results, dt, title=f"IMU Simulation [{args.mode}] — {model_name}")
            plot_noise_comparison(results, dt,
                                 title=f"IMU Noise [{args.mode}] — {model_name}")
        except ImportError:
            print("matplotlib not available — skipping plots")
        except Exception as e:
            print(f"Plotting failed ({e}) — CSV saved successfully")


if __name__ == "__main__":
    main()
