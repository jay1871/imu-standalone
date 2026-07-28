#!/usr/bin/env python3
"""
Allan Variance Validation for IMU Simulator

Computes overlapping Allan deviation from simulated static IMU data and overlays
theoretical NBK model curves for comparison.

Usage:
    python allan_variance.py <csv_file> <config.yaml> [--rate 1000] [--output plot.png]

The theoretical Allan deviation for the NBK model is:
    sigma^2(tau) = N^2/tau + (0.6645*B)^2 + K^2*tau/3

where:
    N = velocity/angular random walk (white noise, slope -1/2)
    B = bias instability coefficient (1/f noise, flat region)
    K = acceleration/rate random walk (random walk, slope +1/2)
"""

import argparse
import sys

import numpy as np
import yaml

try:
    import allantools
except ImportError:
    print("Error: allantools not installed. Run: pip install allantools")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


def compute_theoretical_adev(tau, N, B, K, corr_time):
    """
    Compute theoretical Allan deviation from NBK parameters with exact
    Gauss-Markov bias instability expression.

    Components:
      White noise (N):  sigma^2_N = N^2 / tau
      Bias instability (B): exact 1st-order Gauss-Markov Allan variance:
        sigma^2_B = S_B / (2*beta) * [1 - (2*beta*tau)^-1 * (3 - 4*exp(-beta*tau) + exp(-2*beta*tau))]
        where S_B = 2*B^2*ln(2) / (pi * 0.4365^2 * corr_time), beta = 1/corr_time
      Rate random walk (K): sigma^2_K = K^2 * tau / 3
    """
    beta = 1.0 / corr_time

    # PSD of bias instability (same formula as the C++ model)
    S_B = (2.0 * B**2 * np.log(2)) / (np.pi * 0.4365**2 * corr_time)

    # Exact Gauss-Markov Allan variance for bias instability
    x = beta * tau
    sigma2_B = (S_B / (2.0 * beta)) * (1.0 - (1.0 / (2.0 * x)) * (3.0 - 4.0 * np.exp(-x) + np.exp(-2.0 * x)))

    # White noise
    sigma2_N = N**2 / tau

    # Rate random walk
    sigma2_K = K**2 * tau / 3.0

    sigma2 = sigma2_N + sigma2_B + sigma2_K
    return np.sqrt(sigma2)


def main():
    parser = argparse.ArgumentParser(
        description="Allan Variance Validation for IMU Simulator"
    )
    parser.add_argument("csv_file", help="Path to simulation CSV output")
    parser.add_argument("config_file", help="Path to YAML config used for simulation")
    parser.add_argument("--output", "-o", default="allan_validation_plot.png",
                        help="Output plot filename (default: allan_validation_plot.png)")
    parser.add_argument("--rate", "-r", type=float, default=1000.0,
                        help="Sample rate in Hz (default: 1000)")
    parser.add_argument("--max-tau", type=float, default=None,
                        help="Maximum tau for theoretical curve (default: auto)")
    args = parser.parse_args()

    print(f"Loading data from: {args.csv_file}")
    data = np.loadtxt(args.csv_file, delimiter=",", skiprows=1)
    print(f"  Loaded {len(data)} samples ({len(data)/args.rate/3600:.1f} hours)")

    print(f"Loading config from: {args.config_file}")
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    acc_params = config["imu_simulator"]["model_parameter_settings"]["accelerometer"]
    gyro_params = config["imu_simulator"]["model_parameter_settings"]["gyroscope"]

    rate = args.rate
    max_tau = args.max_tau if args.max_tau else len(data) / rate / 3.0

    # Display unit conversions (from SI)
    # Accel: m/s² → µg:  divide by 9.80665e-6
    # Gyro:  rad/s → deg/hr: multiply by (180/pi) * 3600
    ACCEL_SCALE = 1.0 / 9.80665e-6   # m/s² to µg
    GYRO_SCALE = (180.0 / np.pi) * 3600.0  # rad/s to deg/hr

    sensor_configs = [
        ("Accelerometer", acc_params, 1, "µg", ACCEL_SCALE),
        ("Gyroscope", gyro_params, 4, "deg/hr", GYRO_SCALE),
    ]
    axis_names = ["X", "Y", "Z"]
    axis_colors = ["b", "r", "g"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Allan Deviation Validation — ADIS16607-2 Simulation vs Theory",
                 fontsize=14, fontweight="bold")

    for sensor_idx, (sensor_name, params, col_offset, unit, scale) in enumerate(sensor_configs):
        N_vals = params["N"]
        B_vals = params["B"]
        K_vals = params["K"]
        corr_vals = params["correlation_time"]

        ax = axes[sensor_idx]
        y_min_all = np.inf
        y_max_all = -np.inf

        for axis_idx in range(3):
            col = col_offset + axis_idx
            signal = data[:, col]

            # Remove mean (gravity for accel, earth rate for gyro)
            signal = signal - np.mean(signal)

            print(f"  Computing Allan deviation: {sensor_name} {axis_names[axis_idx]}...")

            # Compute overlapping Allan deviation
            taus_out, adev, adev_err, _ = allantools.oadev(
                signal, rate=rate, data_type="freq", taus="octave"
            )

            # Theoretical curves
            tau_theory = np.logspace(np.log10(1.0 / rate), np.log10(max_tau), 300)
            N = N_vals[axis_idx]
            B = B_vals[axis_idx]
            K = K_vals[axis_idx]
            corr_time = corr_vals[axis_idx]

            adev_theory = compute_theoretical_adev(tau_theory, N, B, K, corr_time)

            color = axis_colors[axis_idx]
            label_sim = f"{axis_names[axis_idx]} simulated"
            label_theory = f"{axis_names[axis_idx]} theory"

            adev_scaled = adev * scale
            adev_theory_scaled = adev_theory * scale

            ax.loglog(taus_out, adev_scaled, color=color, linestyle="-",
                      linewidth=1.5, label=label_sim, alpha=0.9)
            ax.loglog(tau_theory, adev_theory_scaled, color=color, linestyle="--",
                      linewidth=1.2, label=label_theory, alpha=0.7)

            y_min_all = min(y_min_all, np.min(adev_scaled), np.min(adev_theory_scaled))
            y_max_all = max(y_max_all, np.max(adev_scaled), np.max(adev_theory_scaled))

        # Add N/B/K reference lines using first axis params
        N0, B0, K0 = N_vals[0], B_vals[0], K_vals[0]
        tau_theory = np.logspace(np.log10(1.0 / rate), np.log10(max_tau), 300)
        ax.loglog(tau_theory, (N0 / np.sqrt(tau_theory)) * scale, "k:",
                  linewidth=0.8, alpha=0.5, label=f"N (slope -½)")
        ax.loglog(tau_theory, np.full_like(tau_theory, 0.6645 * B0) * scale,
                  "k-.", linewidth=0.8, alpha=0.5, label=f"B (flat)")
        ax.loglog(tau_theory, (K0 * np.sqrt(tau_theory / 3.0)) * scale, "k--",
                  linewidth=0.8, alpha=0.5, label=f"K (slope +½)")

        ax.set_xlabel("Cluster Time τ (s)")
        ax.set_ylabel(f"Allan Deviation ({unit})")
        ax.set_title(f"{sensor_name}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim([0.5, 1e4])
        ax.set_ylim([y_min_all * 0.9, y_max_all * 1.1])

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {args.output}")
    print("\nValidation notes:")
    print("  - Short tau (< 1s): slope should be -1/2, matching N (random walk)")
    print("  - Mid tau (~100-1000s): flat region should match 0.6645*B (bias instability)")
    print("  - Long tau (> 1000s): slope should be +1/2, matching K (rate random walk)")
    print("  - Parameters marked ESTIMATE in YAML (K, correlation_time) may need tuning")


if __name__ == "__main__":
    main()
