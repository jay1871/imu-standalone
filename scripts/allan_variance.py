#!/usr/bin/env python3
"""
Allan Variance Validation for IMU Simulator

Plots Allan deviation results from the allan_validation C++ tool.
Supports two simulator input formats:
  - Single seed: tau, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
  - Multi seed:  tau, acc_x_mean, acc_x_std, ..., gyro_z_mean, gyro_z_std

Optionally overlays theoretical NBK model curves from the YAML configuration
when `--show-theory` is passed.

Usage:
    python allan_variance.py [adev.csv] [config.yaml] [--rate 1000] [--output plot.png] [--gyro-only] [--xl-only] [--show-theory]
"""

import argparse
import os
import sys

import numpy as np
import yaml

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


# Standard axis colors (X, Y, Z): [157, 27, 64; 2, 117, 112; 3, 157, 211] / 255
AXIS_COLORS = [
    (157 / 255.0, 27 / 255.0, 64 / 255.0),
    (2 / 255.0, 117 / 255.0, 112 / 255.0),
    (3 / 255.0, 157 / 255.0, 211 / 255.0),
]


def compute_theoretical_adev(tau, N, B, K, corr_time, B_poles=None, corr_time_poles=None):
    """
    Compute theoretical Allan deviation from NBK parameters with exact
    Gauss-Markov bias instability expression (supports single-pole or multi-pole bank).
    """
    sigma2_N = N**2 / tau
    sigma2_K = K**2 * tau / 3.0

    if B_poles is not None and corr_time_poles is not None:
        sigma2_B = np.zeros_like(tau, dtype=float)
        for b_p, tc_p in zip(B_poles, corr_time_poles):
            beta = 1.0 / tc_p
            S_B = (2.0 * b_p**2 * np.log(2.0)) / (np.pi * 0.4365**2 * tc_p)
            x = beta * tau
            sigma2_B += (S_B / (beta * x)) * (
                1.0 - (1.0 / (2.0 * x)) * (3.0 - 4.0 * np.exp(-x) + np.exp(-2.0 * x))
            )
    else:
        beta = 1.0 / corr_time
        S_B = (2.0 * B**2 * np.log(2.0)) / (np.pi * 0.4365**2 * corr_time)
        x = beta * tau
        sigma2_B = (S_B / (beta * x)) * (
            1.0 - (1.0 / (2.0 * x)) * (3.0 - 4.0 * np.exp(-x) + np.exp(-2.0 * x))
        )

    sigma2 = sigma2_N + sigma2_B + sigma2_K
    return np.sqrt(sigma2)


def style_plot_axes(ax, xlim=None, ylim=None):
    """Apply log-log grid and axis styling."""
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.minorticks_on()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.normpath(os.path.join(script_dir, "..", "config", "adis16607.yaml"))

    parser = argparse.ArgumentParser(
        description="Allan Variance Validation for IMU Simulator"
    )
    parser.add_argument("csv_file", nargs="?", default=None,
                        help="ADEV CSV from allan_validation (single or multi-seed)")
    parser.add_argument("config_file", nargs="?", default=default_config,
                        help="Path to YAML config used for simulation")
    parser.add_argument("--output", "-o", default="allan_validation_plot.png",
                        help="Output plot filename (default: allan_validation_plot.png)")
    parser.add_argument("--rate", "-r", type=float, default=1000.0,
                        help="Sample rate in Hz (default: 1000)")
    parser.add_argument("--max-tau", type=float, default=1e4,
                        help="Maximum tau in seconds for plot (default: 1e4)")
    parser.add_argument("--gyro-only", action="store_true",
                        help="Plot only the gyroscope Allan deviation (single-panel)")
    parser.add_argument("--xl-only", "--accel-only", action="store_true", dest="xl_only",
                        help="Plot only the accelerometer Allan deviation (single-panel)")
    parser.add_argument("--show-theory", action="store_true",
                        help="Overlay theoretical YAML model curve (hidden by default)")
    args = parser.parse_args()

    acc_mean = [None, None, None]
    acc_std = [None, None, None]
    gyro_mean = [None, None, None]
    gyro_std = [None, None, None]
    taus = None

    if args.csv_file and os.path.isfile(args.csv_file):
        print(f"Loading ADEV data from: {args.csv_file}")
        with open(args.csv_file) as f:
            header = f.readline().strip()

        multi_seed = "std" in header
        data = np.loadtxt(args.csv_file, delimiter=",", skiprows=1)
        taus = data[:, 0]

        if multi_seed:
            acc_mean = [data[:, 1], data[:, 3], data[:, 5]]
            acc_std = [data[:, 2], data[:, 4], data[:, 6]]
            gyro_mean = [data[:, 7], data[:, 9], data[:, 11]]
            gyro_std = [data[:, 8], data[:, 10], data[:, 12]]
            print(f"  Multi-seed format: {len(taus)} tau points, mean ± σ")
        else:
            acc_mean = [data[:, 1], data[:, 2], data[:, 3]]
            gyro_mean = [data[:, 4], data[:, 5], data[:, 6]]
            print(f"  Single-seed format: {len(taus)} tau points")

    print(f"Loading config from: {args.config_file}")
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    acc_params = config["imu_simulator"]["model_parameter_settings"]["accelerometer"]
    gyro_params = config["imu_simulator"]["model_parameter_settings"]["gyroscope"]

    ACCEL_SCALE = 1.0 / 9.797464e-6   # m/s² to µg
    GYRO_SCALE = (180.0 / np.pi) * 3600.0  # rad/s to deg/hr

    axis_names = ["X", "Y", "Z"]

    if args.gyro_only:
        fig, ax_gyro = plt.subplots(1, 1, figsize=(9, 6.5), facecolor="white")
        axes = [None, ax_gyro]
        sensor_indices = [1]
    elif args.xl_only:
        fig, ax_xl = plt.subplots(1, 1, figsize=(9, 6.5), facecolor="white")
        axes = [ax_xl, None]
        sensor_indices = [0]
    else:
        fig, axes_arr = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
        axes = [axes_arr[0], axes_arr[1]]
        sensor_indices = [0, 1]

    sensor_configs = [
        ("Accelerometer", acc_params, acc_mean, acc_std, "µg", ACCEL_SCALE, "ADIS16607 accelerometer"),
        ("Gyroscope", gyro_params, gyro_mean, gyro_std, "deg/h", GYRO_SCALE, "ADIS16607-2 gyroscope"),
    ]

    for sensor_idx in sensor_indices:
        sensor_name, params, means, stds, unit, scale, title_prefix = sensor_configs[sensor_idx]
        N_vals = params["N"]
        B_vals = params["B"]
        K_vals = params["K"]
        corr_vals = params["correlation_time"]
        B_poles_all = params.get("B_poles", None)
        corr_poles_all = params.get("correlation_time_poles", None)

        ax = axes[sensor_idx]
        tau_theory = np.logspace(np.log10(0.5), np.log10(args.max_tau), 400)

        for axis_idx in range(3):
            color = AXIS_COLORS[axis_idx]
            ax_label = axis_names[axis_idx]
            b_poles_ax = B_poles_all[axis_idx] if B_poles_all is not None else None
            tc_poles_ax = corr_poles_all[axis_idx] if corr_poles_all is not None else None

            # 1. Plot Simulation curve (if CSV provided)
            if taus is not None and means[axis_idx] is not None:
                adev_scaled = means[axis_idx] * scale
                ax.loglog(taus, adev_scaled, color=color, linestyle="-",
                          linewidth=1.8, marker="o", markersize=3.5,
                          label=f"{ax_label}: sim mean" if stds[axis_idx] is not None else f"{ax_label}: sim")

                if stds[axis_idx] is not None:
                    upper = (means[axis_idx] + stds[axis_idx]) * scale
                    ax.loglog(taus, upper, color=color, linestyle="--",
                              linewidth=1.8, label=f"{ax_label}: sim mean + σ")

            # 2. Plot Theoretical NBK curve from YAML (only if requested or if no sim data is present)
            show_theo = args.show_theory or (taus is None)
            if show_theo:
                theo_adev = compute_theoretical_adev(
                    tau_theory, N_vals[axis_idx], B_vals[axis_idx], K_vals[axis_idx], corr_vals[axis_idx],
                    b_poles_ax, tc_poles_ax
                ) * scale
                theo_style = ":" if taus is not None else "-"
                ax.loglog(tau_theory, theo_adev, color=color, linestyle=theo_style,
                          linewidth=1.5, alpha=0.85, label=f"{ax_label}: YAML model")

        ax.set_xlabel("Integration time τ (s)")
        ax.set_ylabel(f"Allan deviation ({unit})")
        ax.set_title(title_prefix)

        if taus is not None and means[0] is not None:
            mask = (taus >= 0.5) & (taus <= args.max_tau)
            all_mean = np.concatenate([m[mask] * scale for m in means])
            if stds[0] is not None:
                all_upper = np.concatenate([(m[mask] + s[mask]) * scale for m, s in zip(means, stds)])
            else:
                all_upper = all_mean
            ylim = [0.8 * np.min(all_mean), 1.2 * np.max(all_upper)]
        else:
            theo_all = np.concatenate([
                compute_theoretical_adev(
                    tau_theory, N_vals[i], B_vals[i], K_vals[i], corr_vals[i],
                    B_poles_all[i] if B_poles_all is not None else None,
                    corr_poles_all[i] if corr_poles_all is not None else None
                ) * scale
                for i in range(3)
            ])
            ylim = [0.8 * np.min(theo_all), 1.2 * np.max(theo_all)]

        style_plot_axes(ax, xlim=[0.5, args.max_tau], ylim=ylim)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    with open(args.output, "wb") as out_f:
        plt.savefig(out_f, format="png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {args.output}")


if __name__ == "__main__":
    main()
