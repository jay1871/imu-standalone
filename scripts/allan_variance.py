#!/usr/bin/env python3
"""
Allan Variance Validation for IMU Simulator

Plots Allan deviation results from the allan_validation C++ tool.
Supports two input formats:
  - Single seed: tau, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
  - Multi seed:  tau, acc_x_mean, acc_x_std, ..., gyro_z_mean, gyro_z_std

Overlays theoretical NBK model curves and datasheet reference markers.

Usage:
    python allan_variance.py <adev.csv> <config.yaml> [--rate 1000] [--output plot.png]
"""

import argparse
import sys

import numpy as np
import yaml

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


def compute_theoretical_adev(tau, N, B, K, corr_time):
    """
    Compute theoretical Allan deviation from NBK parameters with exact
    Gauss-Markov bias instability expression.
    """
    beta = 1.0 / corr_time

    S_B = (2.0 * B**2 * np.log(2)) / (np.pi * 0.4365**2 * corr_time)

    x = beta * tau
    sigma2_B = (S_B / (2.0 * beta)) * (1.0 - (1.0 / (2.0 * x)) * (3.0 - 4.0 * np.exp(-x) + np.exp(-2.0 * x)))

    sigma2_N = N**2 / tau
    sigma2_K = K**2 * tau / 3.0

    sigma2 = sigma2_N + sigma2_B + sigma2_K
    return np.sqrt(sigma2)


def main():
    parser = argparse.ArgumentParser(
        description="Allan Variance Validation for IMU Simulator"
    )
    parser.add_argument("csv_file", help="ADEV CSV from allan_validation (single or multi-seed)")
    parser.add_argument("config_file", help="Path to YAML config used for simulation")
    parser.add_argument("--output", "-o", default="allan_validation_plot.png",
                        help="Output plot filename (default: allan_validation_plot.png)")
    parser.add_argument("--rate", "-r", type=float, default=1000.0,
                        help="Sample rate in Hz (default: 1000)")
    args = parser.parse_args()

    print(f"Loading ADEV data from: {args.csv_file}")
    with open(args.csv_file) as f:
        header = f.readline().strip()

    multi_seed = "std" in header

    data = np.loadtxt(args.csv_file, delimiter=",", skiprows=1)
    taus = data[:, 0]

    if multi_seed:
        # Format: tau, acc_x_mean, acc_x_std, acc_y_mean, acc_y_std, acc_z_mean, acc_z_std,
        #              gyro_x_mean, gyro_x_std, ...
        acc_mean = [data[:, 1], data[:, 3], data[:, 5]]
        acc_std = [data[:, 2], data[:, 4], data[:, 6]]
        gyro_mean = [data[:, 7], data[:, 9], data[:, 11]]
        gyro_std = [data[:, 8], data[:, 10], data[:, 12]]
        print(f"  Multi-seed format: {len(taus)} tau points, mean ± σ")
    else:
        # Format: tau, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        acc_mean = [data[:, 1], data[:, 2], data[:, 3]]
        acc_std = [None, None, None]
        gyro_mean = [data[:, 4], data[:, 5], data[:, 6]]
        gyro_std = [None, None, None]
        print(f"  Single-seed format: {len(taus)} tau points")

    print(f"Loading config from: {args.config_file}")
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    acc_params = config["imu_simulator"]["model_parameter_settings"]["accelerometer"]
    gyro_params = config["imu_simulator"]["model_parameter_settings"]["gyroscope"]

    ACCEL_SCALE = 1.0 / 9.797464e-6   # m/s² to µg
    GYRO_SCALE = (180.0 / np.pi) * 3600.0  # rad/s to deg/hr

    axis_names = ["X", "Y", "Z"]
    axis_colors = ["b", "r", "g"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Allan Deviation — ADIS16607-2 Simulation vs Theory",
                 fontsize=14, fontweight="bold")

    sensor_configs = [
        ("Accelerometer", acc_params, acc_mean, acc_std, "µg", ACCEL_SCALE),
        ("Gyroscope", gyro_params, gyro_mean, gyro_std, "deg/hr", GYRO_SCALE),
    ]

    for sensor_idx, (sensor_name, params, means, stds, unit, scale) in enumerate(sensor_configs):
        N_vals = params["N"]
        B_vals = params["B"]
        K_vals = params["K"]
        corr_vals = params["correlation_time"]

        ax = axes[sensor_idx]

        for axis_idx in range(3):
            color = axis_colors[axis_idx]
            adev_scaled = means[axis_idx] * scale

            # Main curve
            ax.loglog(taus, adev_scaled, color=color, linestyle="-",
                      linewidth=1.5, label=f"{axis_names[axis_idx]} sim", alpha=0.9)

            # +σ line (multi-seed, no shading)
            if stds[axis_idx] is not None:
                upper = (means[axis_idx] + stds[axis_idx]) * scale
                ax.loglog(taus, upper, color=color, linestyle=":",
                          linewidth=0.8, alpha=0.5, label=f"{axis_names[axis_idx]} +σ")


        # Datasheet reference markers (ADIS16607-2, Rev 0 Draft 2026-08-17)
        if sensor_name == "Gyroscope":
            ds_markers = [
                (10, 18.0, "b", "o", "X DS"),
                (2000, 7.0, "b", "o", None),
                (2000, 10.0, "b", "^", "X DS+σ"),
                (2000, 7.0, "r", "o", "Y DS"),
                (2000, 10.0, "r", "^", "Y DS+σ"),
                (1500, 2.3, "g", "o", "Z DS"),
                (1500, 3.5, "g", "^", "Z DS+σ"),
            ]
            for tau_pt, adev_val, color, marker, label in ds_markers:
                ax.plot(tau_pt, adev_val, marker=marker, color=color,
                        markersize=8, markeredgecolor="black", markeredgewidth=0.5,
                        alpha=0.8, label=label, linestyle="none", zorder=5)

        elif sensor_name == "Accelerometer":
            ds_markers = [
                (1, 100.0, "b", "o", "X DS"),
                (300, 9.0, "b", "o", None),
                (300, 12.0, "b", "^", "X DS+σ"),
                (300, 9.0, "r", "o", "Y DS"),
                (300, 12.0, "r", "^", "Y DS+σ"),
                (1, 90.0, "g", "o", "Z DS"),
                (300, 13.0, "g", "o", None),
                (300, 16.0, "g", "^", "Z DS+σ"),
            ]
            for tau_pt, adev_val, color, marker, label in ds_markers:
                ax.plot(tau_pt, adev_val, marker=marker, color=color,
                        markersize=8, markeredgecolor="black", markeredgewidth=0.5,
                        alpha=0.8, label=label, linestyle="none", zorder=5)

        ax.set_xlabel("Cluster Time τ (s)")
        ax.set_ylabel(f"Allan Deviation ({unit})")
        ax.set_title(f"{sensor_name}")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {args.output}")


if __name__ == "__main__":
    main()
