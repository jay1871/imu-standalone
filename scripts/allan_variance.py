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


def compute_theoretical_adev(tau, N, B, K):
    """
    Compute theoretical Allan deviation from NBK parameters.

    sigma^2(tau) = N^2/tau + (0.6645*B)^2 + K^2*tau/3
    """
    sigma2 = (N**2) / tau + (0.6645 * B) ** 2 + (K**2 * tau) / 3.0
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

    # Set up the plot: 2 rows (accel, gyro) x 3 columns (X, Y, Z)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Allan Deviation Validation — ADIS16607-2 Simulation vs Theory",
                 fontsize=14, fontweight="bold")

    sensor_configs = [
        ("Accelerometer", acc_params, 1, "m/s²"),
        ("Gyroscope", gyro_params, 4, "rad/s"),
    ]
    axis_names = ["X", "Y", "Z"]

    for sensor_idx, (sensor_name, params, col_offset, unit) in enumerate(sensor_configs):
        N_vals = params["N"]
        B_vals = params["B"]
        K_vals = params["K"]

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

            adev_theory = compute_theoretical_adev(tau_theory, N, B, K)
            adev_N = N / np.sqrt(tau_theory)
            adev_B = np.full_like(tau_theory, 0.6645 * B)
            adev_K = K * np.sqrt(tau_theory / 3.0)

            # Plot
            ax = axes[sensor_idx, axis_idx]
            ax.loglog(taus_out, adev, "b-", linewidth=1.5, label="Simulated", alpha=0.9)
            ax.loglog(tau_theory, adev_theory, "r--", linewidth=1.5,
                      label="Theory (N+B+K)")
            ax.loglog(tau_theory, adev_N, "g:", linewidth=1.0, alpha=0.7,
                      label=f"N={N:.2e} (slope -½)")
            ax.loglog(tau_theory, adev_B, "m:", linewidth=1.0, alpha=0.7,
                      label=f"B={B:.2e} (flat)")
            ax.loglog(tau_theory, adev_K, "c:", linewidth=1.0, alpha=0.7,
                      label=f"K={K:.2e} (slope +½)")

            ax.set_xlabel("Cluster Time τ (s)")
            ax.set_ylabel(f"Allan Deviation ({unit})")
            ax.set_title(f"{sensor_name} — {axis_names[axis_idx]}-Axis")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xlim([1.0 / rate, max_tau])

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
