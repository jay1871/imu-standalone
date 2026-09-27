#!/usr/bin/env python3
"""Plot IMU simulation results from CSV output.

Usage:
  # Plot from simulate_imu.py output
  python plot_results.py imu_sim_plugin.csv

  # Compare two runs (e.g., plugin vs mjx)
  python plot_results.py imu_sim_plugin.csv imu_sim_mjx.csv

  # Plot specific time range
  python plot_results.py imu_sim_plugin.csv --tstart 1.0 --tend 3.0

  # Plot only noise (noisy - true)
  python plot_results.py imu_sim_plugin.csv --noise-only

  # Plot power spectral density
  python plot_results.py imu_sim_plugin.csv --psd
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    """Load simulation CSV into a dict."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {
        "time": data["time"],
        "acc_true": np.column_stack([data["acc_true_x"], data["acc_true_y"], data["acc_true_z"]]),
        "gyro_true": np.column_stack([data["gyro_true_x"], data["gyro_true_y"], data["gyro_true_z"]]),
        "acc_noisy": np.column_stack([data["acc_noisy_x"], data["acc_noisy_y"], data["acc_noisy_z"]]),
        "gyro_noisy": np.column_stack([data["gyro_noisy_x"], data["gyro_noisy_y"], data["gyro_noisy_z"]]),
    }


def trim_time(d, tstart, tend):
    """Trim data to time window."""
    mask = (d["time"] >= tstart) & (d["time"] <= tend)
    return {k: v[mask] if isinstance(v, np.ndarray) and v.ndim > 0 else v
            for k, v in d.items()}


def plot_true_vs_noisy(datasets, labels, title="IMU Simulation"):
    """6-panel plot: true vs noisy for each axis."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=13)
    axis_names = ["X", "Y", "Z"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for di, (d, label) in enumerate(zip(datasets, labels)):
        t = d["time"]
        c = colors[di % len(colors)]
        for ax_i in range(3):
            axes[0, ax_i].plot(t, d["acc_true"][:, ax_i], "-", color=c,
                               alpha=0.4, linewidth=0.5, label=f"{label} true")
            axes[0, ax_i].plot(t, d["acc_noisy"][:, ax_i], "-", color=c,
                               alpha=0.7, linewidth=0.3, label=f"{label} noisy")
            axes[1, ax_i].plot(t, d["gyro_true"][:, ax_i], "-", color=c,
                               alpha=0.4, linewidth=0.5)
            axes[1, ax_i].plot(t, d["gyro_noisy"][:, ax_i], "-", color=c,
                               alpha=0.7, linewidth=0.3)

    for ax_i in range(3):
        axes[0, ax_i].set_ylabel(f"Accel {axis_names[ax_i]} (m/s²)")
        axes[0, ax_i].grid(True, alpha=0.3)
        axes[0, ax_i].legend(fontsize=7, loc="upper right")
        axes[1, ax_i].set_ylabel(f"Gyro {axis_names[ax_i]} (rad/s)")
        axes[1, ax_i].set_xlabel("Time (s)")
        axes[1, ax_i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_noise_only(datasets, labels, title="IMU Noise (Noisy - True)"):
    """Plot just the error component."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=13)
    axis_names = ["X", "Y", "Z"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for di, (d, label) in enumerate(zip(datasets, labels)):
        t = d["time"]
        c = colors[di % len(colors)]
        acc_err = d["acc_noisy"] - d["acc_true"]
        gyro_err = d["gyro_noisy"] - d["gyro_true"]
        for ax_i in range(3):
            axes[0, ax_i].plot(t, acc_err[:, ax_i], "-", color=c,
                               alpha=0.6, linewidth=0.3, label=label)
            axes[1, ax_i].plot(t, gyro_err[:, ax_i], "-", color=c,
                               alpha=0.6, linewidth=0.3, label=label)

    for ax_i in range(3):
        axes[0, ax_i].set_ylabel(f"Accel err {axis_names[ax_i]} (m/s²)")
        axes[0, ax_i].grid(True, alpha=0.3)
        axes[0, ax_i].legend(fontsize=8)
        axes[1, ax_i].set_ylabel(f"Gyro err {axis_names[ax_i]} (rad/s)")
        axes[1, ax_i].set_xlabel("Time (s)")
        axes[1, ax_i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_psd(datasets, labels, title="Power Spectral Density"):
    """Plot PSD of the noise component."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=13)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for di, (d, label) in enumerate(zip(datasets, labels)):
        t = d["time"]
        dt = t[1] - t[0] if len(t) > 1 else 0.001
        fs = 1.0 / dt
        n = len(t)

        acc_err = d["acc_noisy"] - d["acc_true"]
        gyro_err = d["gyro_noisy"] - d["gyro_true"]

        # Average PSD across 3 axes
        from scipy.signal import welch
        nperseg = min(n // 4, 4096)

        f_a, psd_a = welch(acc_err, fs=fs, nperseg=nperseg, axis=0)
        f_g, psd_g = welch(gyro_err, fs=fs, nperseg=nperseg, axis=0)

        c = colors[di % len(colors)]
        axes[0].loglog(f_a, psd_a.mean(axis=1), color=c, alpha=0.8, label=label)
        axes[1].loglog(f_g, psd_g.mean(axis=1), color=c, alpha=0.8, label=label)

    axes[0].set_ylabel("Accel PSD (m/s²)²/Hz")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which="both")
    axes[1].set_ylabel("Gyro PSD (rad/s)²/Hz")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


def plot_statistics(datasets, labels):
    """Print summary statistics of the noise."""
    print("\n" + "=" * 70)
    print("NOISE STATISTICS")
    print("=" * 70)
    axis_names = ["X", "Y", "Z"]

    for d, label in zip(datasets, labels):
        acc_err = d["acc_noisy"] - d["acc_true"]
        gyro_err = d["gyro_noisy"] - d["gyro_true"]
        duration = d["time"][-1] - d["time"][0]

        print(f"\n--- {label} ({duration:.1f}s, {len(d['time'])} samples) ---")
        print(f"  {'Axis':<6s} {'Acc mean':>12s} {'Acc std':>12s} {'Gyro mean':>12s} {'Gyro std':>12s}")
        for i, ax in enumerate(axis_names):
            print(f"  {ax:<6s} {acc_err[:, i].mean():>12.2e} {acc_err[:, i].std():>12.2e} "
                  f"{gyro_err[:, i].mean():>12.2e} {gyro_err[:, i].std():>12.2e}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Plot IMU simulation results")
    parser.add_argument("csvfiles", nargs="+", help="CSV file(s) from simulate_imu.py")
    parser.add_argument("--tstart", type=float, default=None, help="Start time (s)")
    parser.add_argument("--tend", type=float, default=None, help="End time (s)")
    parser.add_argument("--noise-only", action="store_true", help="Plot only noise component")
    parser.add_argument("--psd", action="store_true", help="Plot power spectral density")
    parser.add_argument("--stats", action="store_true", help="Print noise statistics")
    parser.add_argument("--save", type=str, default=None, help="Save figure to file (png/pdf)")
    parser.add_argument("--no-show", action="store_true", help="Don't display (for headless)")

    args = parser.parse_args()

    datasets = []
    labels = []
    for f in args.csvfiles:
        datasets.append(load_csv(f))
        labels.append(Path(f).stem)

    # Trim time
    if args.tstart is not None or args.tend is not None:
        tstart = args.tstart or 0
        tend = args.tend or datasets[0]["time"][-1]
        datasets = [trim_time(d, tstart, tend) for d in datasets]

    # Always print stats
    if args.stats or True:
        plot_statistics(datasets, labels)

    # Generate plots
    figs = []
    if args.noise_only:
        figs.append(plot_noise_only(datasets, labels))
    elif args.psd:
        figs.append(plot_psd(datasets, labels))
    else:
        figs.append(plot_true_vs_noisy(datasets, labels))
        figs.append(plot_noise_only(datasets, labels))

    if args.save:
        for i, fig in enumerate(figs):
            suffix = f"_{i}" if len(figs) > 1 else ""
            path = Path(args.save).stem + suffix + Path(args.save).suffix
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
