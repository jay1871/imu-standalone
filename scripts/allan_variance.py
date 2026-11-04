#!/usr/bin/env python3
"""
Allan Variance Validation for IMU Simulator

Plots Allan deviation results from the allan_validation C++ tool and/or
optional real-world hardware characterization CSV files.
Supports two simulator input formats:
  - Single seed: tau, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
  - Multi seed:  tau, acc_x_mean, acc_x_std, ..., gyro_z_mean, gyro_z_std

Overlays theoretical NBK model curves and optional real-world reference curves.

Usage:
    python allan_variance.py [adev.csv] [config.yaml] [--rate 1000] [--output plot.png] [--measured-dir /path/to/testdata] [--gyro-only] [--xl-only]
"""

import argparse
import csv
import os
import sys

import numpy as np
import yaml

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


# Set ENABLE_MEASURED_DATASET = True (or pass --measured-dir <path>) to load and overlay
# real-world hardware characterization CSV datasets.
# By default (False), this script does not look for any external measurement files.
ENABLE_MEASURED_DATASET = False

# Standard axis colors (X, Y, Z): [157, 27, 64; 2, 117, 112; 3, 157, 211] / 255
AXIS_COLORS = [
    (157 / 255.0, 27 / 255.0, 64 / 255.0),
    (2 / 255.0, 117 / 255.0, 112 / 255.0),
    (3 / 255.0, 157 / 255.0, 211 / 255.0),
]

MEASURED_GYRO_FILES_AND_EXCLUDED = [
    ("Lot269_Gyro_AVAR_2026_03_30_15_35_07_readable.csv", [6, 7]),
    ("Lot9960_Gyro_AVAR_2026_03_30_17_34_26_readable.csv", [2, 5, 6, 7, 8, 9, 12, 13, 15]),
    ("Lot841_450dps_Gyro_2025_10_30_16_03_10_readable.csv", [0, 3, 6, 14]),
]

MEASURED_XL_FILES_AND_EXCLUDED = [
    ("Lot269_XL_XLspoiler_AVAR_2026_07_27_14_43_34_readable.csv", []),
    ("Lot9960_XL_XLspoilerON_AVAR_2026_07_31_14_29_53_readable.csv", []),
    ("Lot841_XLspoiler_AVAR_2026_07_20_13_25_00_readable.csv", [3]),
]


def compute_theoretical_adev(tau, N, B, K, corr_time):
    """
    Compute theoretical Allan deviation from NBK parameters with exact
    Gauss-Markov bias instability expression.
    """
    beta = 1.0 / corr_time

    S_B = (2.0 * B**2 * np.log(2.0)) / (np.pi * 0.4365**2 * corr_time)

    x = beta * tau
    sigma2_B = (S_B / (beta * x)) * (
        1.0 - (1.0 / (2.0 * x)) * (3.0 - 4.0 * np.exp(-x) + np.exp(-2.0 * x))
    )

    sigma2_N = N**2 / tau
    sigma2_K = K**2 * tau / 3.0

    sigma2 = sigma2_N + sigma2_B + sigma2_K
    return np.sqrt(sigma2)


def _load_measured_csv_group(measured_dir, files_and_excluded, unit_str):
    if not measured_dir or not os.path.isdir(measured_dir):
        return None

    axis_names = ["X", "Y", "Z"]
    taus_list = []
    adev_by_axis = [[], [], []]

    for fname, excluded in files_and_excluded:
        fpath = os.path.join(measured_dir, fname)
        if not os.path.isfile(fpath):
            return None
        with open(fpath, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        tau = np.array([float(r["tau (s)"]) for r in rows])
        included_duts = [d for d in range(16) if d not in excluded]
        for _ in included_duts:
            taus_list.append(tau)
        for ax_idx, ax_name in enumerate(axis_names):
            for d in included_duts:
                col = f"{ax_name} DUT{d} ADEV ({unit_str})"
                vals = np.array([float(r[col]) for r in rows])
                adev_by_axis[ax_idx].append(vals)

    tau_by_dut = np.column_stack(taus_list)
    tau_mean = np.mean(tau_by_dut, axis=1)
    dut_count = tau_by_dut.shape[1]

    mean_adev = []
    std_adev = []
    for ax_idx in range(3):
        mat = np.column_stack(adev_by_axis[ax_idx])
        mean_adev.append(np.mean(mat, axis=1))
        std_adev.append(np.std(mat, axis=1, ddof=1))

    return tau_mean, mean_adev, std_adev, dut_count


def load_measured_gyro_data(measured_dir):
    """
    Load real-world gyroscope readable CSV files:
    33 selected DUTs across 3 lots, returning (tau_mean, mean_adev, std_adev, dut_count).
    """
    return _load_measured_csv_group(measured_dir, MEASURED_GYRO_FILES_AND_EXCLUDED, "deg/h")


def load_measured_xl_data(measured_dir):
    """
    Load real-world accelerometer readable CSV files:
    47 selected DUTs across 3 lots, returning (tau_mean, mean_adev, std_adev, dut_count).
    """
    return _load_measured_csv_group(measured_dir, MEASURED_XL_FILES_AND_EXCLUDED, "ug")


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
    default_measured_dir = (
        os.path.normpath(os.path.join(script_dir, "..", "..", "..", "fromBU", "testdata"))
        if ENABLE_MEASURED_DATASET
        else None
    )

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
    parser.add_argument("--measured-dir", "--bu-dir", dest="measured_dir", default=default_measured_dir,
                        help="Optional path to real-world measurement CSV folder (disabled by default)")
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

    measured_gyro_data = load_measured_gyro_data(args.measured_dir)
    if measured_gyro_data is not None:
        print(f"Loaded measured gyro data: {measured_gyro_data[3]} DUTs, {len(measured_gyro_data[0])} tau values from {args.measured_dir}")

    measured_xl_data = load_measured_xl_data(args.measured_dir)
    if measured_xl_data is not None:
        print(f"Loaded measured accelerometer data: {measured_xl_data[3]} DUTs, {len(measured_xl_data[0])} tau values from {args.measured_dir}")

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
        ("Accelerometer", acc_params, acc_mean, acc_std, "µg", ACCEL_SCALE, measured_xl_data, "ADIS16607 accelerometer"),
        ("Gyroscope", gyro_params, gyro_mean, gyro_std, "deg/h", GYRO_SCALE, measured_gyro_data, "ADIS16607-2 gyroscope"),
    ]

    for sensor_idx in sensor_indices:
        sensor_name, params, means, stds, unit, scale, measured_sensor_data, title_prefix = sensor_configs[sensor_idx]
        N_vals = params["N"]
        B_vals = params["B"]
        K_vals = params["K"]
        corr_vals = params["correlation_time"]

        ax = axes[sensor_idx]
        tau_theory = np.logspace(np.log10(0.5), np.log10(args.max_tau), 400)

        for axis_idx in range(3):
            color = AXIS_COLORS[axis_idx]
            ax_label = axis_names[axis_idx]

            # 1. If real-world measured data is available for this sensor, plot reference curves
            if measured_sensor_data is not None:
                m_tau, m_mean, m_std, _ = measured_sensor_data
                ax.loglog(m_tau, m_mean[axis_idx], color=color, linestyle="-",
                          linewidth=1.8, label=f"{ax_label}: mean (measured)" if taus is not None else f"{ax_label}: mean")
                ax.loglog(m_tau, m_mean[axis_idx] + m_std[axis_idx], color=color, linestyle="--",
                          linewidth=1.8, label=f"{ax_label}: mean + σ (measured)" if taus is not None else f"{ax_label}: mean + σ")

            # 2. Plot Simulation curve (if CSV provided)
            if taus is not None and means[axis_idx] is not None:
                adev_scaled = means[axis_idx] * scale
                sim_style = "-." if measured_sensor_data is not None else "-"
                ax.loglog(taus, adev_scaled, color=color, linestyle=sim_style,
                          linewidth=1.8, marker="o", markersize=3.5,
                          label=f"{ax_label}: sim mean" if stds[axis_idx] is not None else f"{ax_label}: sim")

                if stds[axis_idx] is not None:
                    upper = (means[axis_idx] + stds[axis_idx]) * scale
                    ax.loglog(taus, upper, color=color, linestyle=":",
                              linewidth=1.5, label=f"{ax_label}: sim mean + σ")

            # 3. Plot Theoretical NBK curve from YAML (only if requested or if neither measured nor sim data is present)
            show_theo = args.show_theory or (taus is None and measured_sensor_data is None)
            if show_theo:
                theo_adev = compute_theoretical_adev(
                    tau_theory, N_vals[axis_idx], B_vals[axis_idx], K_vals[axis_idx], corr_vals[axis_idx]
                ) * scale
                theo_style = ":" if (measured_sensor_data is not None or taus is not None) else "-"
                ax.loglog(tau_theory, theo_adev, color=color, linestyle=theo_style,
                          linewidth=1.5, alpha=0.85, label=f"{ax_label}: YAML model")

        ax.set_xlabel("Integration time τ (s)")
        ax.set_ylabel(f"Allan deviation ({unit})")
        if measured_sensor_data is not None:
            m_tau, m_mean, m_std, m_dut_count = measured_sensor_data
            ax.set_title(f"{title_prefix} - {m_dut_count} DUTs")
            all_mean = np.concatenate(m_mean)
            all_upper = np.concatenate([m + s for m, s in zip(m_mean, m_std)])
            ylim = [0.8 * np.min(all_mean), 1.2 * np.max(all_upper)]
            style_plot_axes(ax, xlim=[0.5, args.max_tau], ylim=ylim)
        else:
            ax.set_title(title_prefix)
            theo_all = np.concatenate([
                compute_theoretical_adev(tau_theory, N_vals[i], B_vals[i], K_vals[i], corr_vals[i]) * scale
                for i in range(3)
            ])
            ylim = [0.6 * np.min(theo_all), 1.5 * np.max(theo_all)]
            style_plot_axes(ax, xlim=[0.5, args.max_tau], ylim=ylim)
        ax.legend(loc="upper right", fontsize=8 if (taus is not None and measured_sensor_data is not None) else 9)

    plt.tight_layout()
    with open(args.output, "wb") as out_f:
        plt.savefig(out_f, format="png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {args.output}")


if __name__ == "__main__":
    main()
