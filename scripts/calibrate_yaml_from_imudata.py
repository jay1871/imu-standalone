#!/usr/bin/env python3
"""
Calibrate & Validate IMU Simulator YAML Configuration Against Hardware Data (imudata/)

Checks collected hardware Allan deviation CSV datasets in `imudata/` (and optional
MATLAB exclusion scripts `plot_gyro_avar.m` / `plot_xl_avar.m`) against a target
simulator YAML configuration file (e.g., `config/adis16607.yaml`).

Features:
  1. Loads per-DUT hardware Allan deviation curves (Accelerometer in ug, Gyroscope in deg/h)
     from `imudata/`, automatically parsing MATLAB exclusion lists if present.
  2. Evaluates the current YAML parameters (N, B, K, correlation_time, B_poles,
     correlation_time_poles, and DUT spread N_std, B_std, K_std, correlation_time_std)
     against the measured ensemble mean and mean + sigma curves.
  3. Reports per-axis RMS relative error (%), Max relative error (%), log-log convexity
     (hump detection), and 1-sigma upper-envelope error (%).
  4. If any axis exceeds `--tolerance-pct` (default: 1.0%) or `--force-update` is passed
     (and `--check-only` is not set), fits optimal multi-pole Gauss-Markov parameters
     (Option B), single-pole fallback parameters, and DUT-to-DUT spread parameters,
     then updates the YAML file in-place while preserving all comments and formatting.

Usage:
    python scripts/calibrate_yaml_from_imudata.py
    python scripts/calibrate_yaml_from_imudata.py --check-only
    python scripts/calibrate_yaml_from_imudata.py --force-update
"""

import argparse
import csv
import glob
import os
import re
import shutil
import sys

import numpy as np
import yaml
from scipy.optimize import minimize, nnls

# Unit conversions to SI
ACCEL_SCALE = 1.0 / 9.797464e-6          # m/s^2 -> ug
GYRO_SCALE = (180.0 / np.pi) * 3600.0    # rad/s -> deg/h

AXIS_NAMES = ["X", "Y", "Z"]

DEFAULT_GYRO_FILES_AND_EXCLUDED = [
    ("Lot269_Gyro_AVAR_2026_03_30_15_35_07_readable.csv", [6, 7]),
    ("Lot9960_Gyro_AVAR_2026_03_30_17_34_26_readable.csv", [2, 5, 6, 7, 8, 9, 12, 13, 15]),
    ("Lot841_450dps_Gyro_2025_10_30_16_03_10_readable.csv", [0, 3, 6, 14]),
]

DEFAULT_XL_FILES_AND_EXCLUDED = [
    ("Lot269_XL_XLspoiler_AVAR_2026_07_27_14_43_34_readable.csv", []),
    ("Lot9960_XL_XLspoilerON_AVAR_2026_07_31_14_29_53_readable.csv", []),
    ("Lot841_XLspoiler_AVAR_2026_07_20_13_25_00_readable.csv", [3]),
]


def parse_matlab_dataset_spec(matlab_script_path):
    """
    Parse `fileNames` and `excludedDuts` from `plot_gyro_avar.m` or `plot_xl_avar.m`
    if present in `imudata/`. Returns a list of (filename, excluded_dut_list) tuples,
    or None if parsing fails.
    """
    if not os.path.isfile(matlab_script_path):
        return None

    with open(matlab_script_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    fn_match = re.search(r"fileNames\s*=\s*\[(.*?)\];", content, re.DOTALL)
    ex_match = re.search(r"excludedDuts\s*=\s*\{(.*?)\};", content, re.DOTALL)
    if not fn_match or not ex_match:
        return None

    file_names = re.findall(r'"([^"]+\.csv)"', fn_match.group(1))
    raw_ex_parts = ex_match.group(1).split(";")
    if len(file_names) != len(raw_ex_parts):
        return None

    spec = []
    for fname, part in zip(file_names, raw_ex_parts):
        nums = [int(x) for x in re.findall(r"\d+", part)]
        spec.append((fname, nums))
    return spec


def load_imudata_sensor_group(imudata_dir, matlab_script_name, default_spec, unit_str, csv_glob_pattern):
    """
    Load hardware Allan deviation CSV files for a sensor group (accelerometer or gyroscope).
    Returns a dict with:
      - tau: (N_tau,) mean tau vector in seconds
      - mean_adev_si: list of 3 arrays (X, Y, Z) in SI units
      - std_adev_si: list of 3 arrays (X, Y, Z) in SI units
      - dut_adev_si: list of 3 arrays of shape (N_tau, N_duts) in SI units
      - dut_count: int total included DUTs
      - files_used: list of (filename, included_count, excluded_list)
    """
    scale = ACCEL_SCALE if unit_str == "ug" else GYRO_SCALE

    matlab_path = os.path.join(imudata_dir, matlab_script_name)
    spec = parse_matlab_dataset_spec(matlab_path)
    if spec is None:
        if all(os.path.isfile(os.path.join(imudata_dir, fn)) for fn, _ in default_spec):
            spec = default_spec
        else:
            discovered = sorted(glob.glob(os.path.join(imudata_dir, csv_glob_pattern)))
            spec = [(os.path.basename(p), []) for p in discovered]

    if not spec:
        raise FileNotFoundError(f"No CSV files found in {imudata_dir} for unit '{unit_str}'")

    taus_list = []
    adev_by_axis = [[], [], []]
    files_used = []

    for fname, excluded in spec:
        fpath = os.path.join(imudata_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Expected measurement CSV file not found: {fpath}")

        with open(fpath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)

        tau = np.array([float(r["tau (s)"]) for r in rows], dtype=float)

        dut_indices = sorted({
            int(m.group(1))
            for h in headers
            for m in [re.match(rf"X DUT(\d+) ADEV \({re.escape(unit_str)}\)", h)]
            if m
        })
        included_duts = [d for d in dut_indices if d not in excluded]
        files_used.append((fname, len(included_duts), excluded))

        for _ in included_duts:
            taus_list.append(tau)
        for ax_idx, ax_name in enumerate(AXIS_NAMES):
            for d in included_duts:
                col = f"{ax_name} DUT{d} ADEV ({unit_str})"
                vals = np.array([float(r[col]) for r in rows], dtype=float) / scale
                adev_by_axis[ax_idx].append(vals)

    tau_by_dut = np.column_stack(taus_list)
    tau_mean = np.mean(tau_by_dut, axis=1)
    dut_count = tau_by_dut.shape[1]

    mean_adev_si = []
    std_adev_si = []
    dut_adev_si = []
    for ax_idx in range(3):
        mat = np.column_stack(adev_by_axis[ax_idx])
        dut_adev_si.append(mat)
        mean_adev_si.append(np.mean(mat, axis=1))
        std_adev_si.append(np.std(mat, axis=1, ddof=1))

    return {
        "tau": tau_mean,
        "mean_adev_si": mean_adev_si,
        "std_adev_si": std_adev_si,
        "dut_adev_si": dut_adev_si,
        "dut_count": dut_count,
        "files_used": files_used,
    }


def gm_unit_basis(tau, corr_time):
    """
    Theoretical Allan variance for a 1st-order Gauss-Markov process with unit B=1.0
    and correlation time `corr_time`.
    """
    beta = 1.0 / corr_time
    S_B_unit = (2.0 * np.log(2.0)) / (np.pi * 0.4365**2 * corr_time)
    x = beta * tau
    return (S_B_unit / (beta * x)) * (
        1.0 - (1.0 / (2.0 * x)) * (3.0 - 4.0 * np.exp(-x) + np.exp(-2.0 * x))
    )


def compute_theoretical_adev(tau, N, B, K, corr_time, B_poles=None, corr_time_poles=None):
    """
    Compute theoretical Allan deviation in SI units from NBK parameters
    (supports single-pole or multi-pole Gauss-Markov bank).
    """
    sigma2_N = (N**2) / tau
    sigma2_K = (K**2) * tau / 3.0

    if B_poles is not None and corr_time_poles is not None and len(B_poles) > 0:
        sigma2_B = np.zeros_like(tau, dtype=float)
        for b_p, tc_p in zip(B_poles, corr_time_poles):
            if b_p > 0.0 and tc_p > 0.0:
                sigma2_B += (b_p**2) * gm_unit_basis(tau, tc_p)
    else:
        sigma2_B = (B**2) * gm_unit_basis(tau, corr_time)

    return np.sqrt(sigma2_N + sigma2_B + sigma2_K)


def compute_theoretical_upper_adev(
    tau, N, B, K, corr_time, N_std, B_std, K_std, B_poles=None, corr_time_poles=None
):
    """
    Compute theoretical 1-sigma upper envelope (`mean + sigma`) Allan deviation
    by evaluating the model at (N + N_std, B + B_std, K + K_std), where multi-pole
    amplitudes scale proportionally with (1 + B_std / B) just as in `sampleDutParams()`.
    """
    N_up = N + N_std
    B_up = B + B_std
    K_up = K + K_std
    if B_poles is not None and B > 0.0:
        scale_b = B_up / B
        B_poles_up = [bp * scale_b for bp in B_poles]
    else:
        B_poles_up = B_poles
    return compute_theoretical_adev(tau, N_up, B_up, K_up, corr_time, B_poles_up, corr_time_poles)


def evaluate_axis_metrics(tau, y_mean_si, y_std_si, params, ax_idx):
    """
    Evaluate accuracy & convexity metrics for one axis of the YAML config against measured data.
    """
    N = float(params["N"][ax_idx])
    B = float(params["B"][ax_idx])
    K = float(params["K"][ax_idx])
    corr_time = float(params["correlation_time"][ax_idx])

    B_poles_all = params.get("B_poles", None)
    corr_poles_all = params.get("correlation_time_poles", None)
    b_poles = [float(x) for x in B_poles_all[ax_idx]] if B_poles_all is not None else None
    tc_poles = [float(x) for x in corr_poles_all[ax_idx]] if corr_poles_all is not None else None

    N_std = float(params.get("N_std", [0.0, 0.0, 0.0])[ax_idx])
    B_std = float(params.get("B_std", [0.0, 0.0, 0.0])[ax_idx])
    K_std = float(params.get("K_std", [0.0, 0.0, 0.0])[ax_idx])

    pred_mean = compute_theoretical_adev(tau, N, B, K, corr_time, b_poles, tc_poles)
    rel_diff = (pred_mean - y_mean_si) / y_mean_si
    rms_err_pct = float(np.sqrt(np.mean(rel_diff**2)) * 100.0)
    max_idx = int(np.argmax(np.abs(rel_diff)))
    max_err_pct = float(np.abs(rel_diff[max_idx]) * 100.0)
    max_err_tau = float(tau[max_idx])

    tau_dense = np.logspace(np.log10(np.min(tau)), np.log10(np.max(tau)), 300)
    log_tau_dense = np.log(tau_dense)
    pred_dense = compute_theoretical_adev(tau_dense, N, B, K, corr_time, b_poles, tc_poles)
    d1 = np.gradient(np.log(pred_dense), log_tau_dense)
    d2 = np.gradient(d1, log_tau_dense)
    min_d2 = float(np.min(d2))
    is_convex = min_d2 >= -1e-3

    y_upper = y_mean_si + y_std_si
    pred_upper = compute_theoretical_upper_adev(
        tau, N, B, K, corr_time, N_std, B_std, K_std, b_poles, tc_poles
    )
    upper_rel_diff = (pred_upper - y_upper) / y_upper
    upper_rms_err_pct = float(np.sqrt(np.mean(upper_rel_diff**2)) * 100.0)

    return {
        "rms_err_pct": rms_err_pct,
        "max_err_pct": max_err_pct,
        "max_err_tau": max_err_tau,
        "min_d2": min_d2,
        "is_convex": is_convex,
        "upper_rms_err_pct": upper_rms_err_pct,
        "pred_mean": pred_mean,
        "pred_upper": pred_upper,
    }


def fit_axis_multipole_and_spread(tau, y_mean_si, y_std_si, dut_mat_si, n_poles=3):
    """
    Fit optimal multi-pole NBK parameters (N, K, B_poles, correlation_time_poles),
    equivalent single-pole fallback (B, correlation_time), and DUT spread parameters
    (N_std, B_std, K_std, correlation_time_std) for a single axis.
    """
    y_var = y_mean_si**2

    tc_starts = [
        [20.0, 180.0, 1600.0],
        [35.0, 180.0, 900.0],
        [15.0, 140.0, 2000.0],
    ]
    if n_poles != 3:
        tc_starts = [np.geomspace(20.0, 1800.0, n_poles).tolist()]

    def loss(log_tcs):
        tcs = np.sort(np.exp(log_tcs))
        if np.any(tcs < 2.0) or np.any(tcs > 20000.0):
            return 1e9
        if np.any(np.diff(np.log10(tcs)) < 0.35):
            return 1e9
        A = np.column_stack([1.0 / tau] + [gm_unit_basis(tau, tc) for tc in tcs] + [tau / 3.0])
        Aw = A / y_var[:, None]
        bw = np.ones_like(y_var)
        coeffs, _ = nnls(Aw, bw)
        pred_adev = np.sqrt(A @ coeffs)
        rel_err = np.log(pred_adev / y_mean_si)

        tau_dense = np.logspace(np.log10(np.min(tau)), np.log10(np.max(tau)), 200)
        A_dense = np.column_stack(
            [1.0 / tau_dense] + [gm_unit_basis(tau_dense, tc) for tc in tcs] + [tau_dense / 3.0]
        )
        pred_dense = np.sqrt(A_dense @ coeffs)
        d1 = np.gradient(np.log(pred_dense), np.log(tau_dense))
        d2 = np.gradient(d1, np.log(tau_dense))
        non_convex_penalty = np.sum(np.minimum(0.0, d2) ** 2) * 50.0
        return float(np.mean(rel_err**2) + non_convex_penalty)

    best_res = None
    for tc_init in tc_starts:
        res = minimize(
            loss,
            np.log(tc_init),
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-9},
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    tcs = np.sort(np.exp(best_res.x))
    A = np.column_stack([1.0 / tau] + [gm_unit_basis(tau, tc) for tc in tcs] + [tau / 3.0])
    coeffs, _ = nnls(A / y_var[:, None], np.ones_like(y_var))

    N_fit = float(np.sqrt(coeffs[0]))
    B_poles_fit = [float(np.sqrt(c)) for c in coeffs[1:-1]]
    Tc_poles_fit = [float(tc) for tc in tcs]
    K_fit = float(np.sqrt(coeffs[-1]))

    min_idx = int(np.argmin(y_mean_si))
    tau_min = float(tau[min_idx])
    corr_time_single = round(float(np.clip(tau_min / 1.89, 50.0, 600.0)), 1)
    sigma2_rem = np.maximum(0.0, y_var - (N_fit**2) / tau - (K_fit**2) * tau / 3.0)
    gm_single_vec = gm_unit_basis(tau, corr_time_single)
    b2_single = float(np.sum(gm_single_vec * sigma2_rem / y_var) / np.sum((gm_single_vec**2) / y_var))
    B_single = float(np.sqrt(max(b2_single, 1e-16)))

    n_duts = dut_mat_si.shape[1]
    dut_Ns = []
    dut_Ks = []
    dut_tau_mins = []
    for d_idx in range(n_duts):
        d_curve = dut_mat_si[:, d_idx]
        d_var = d_curve**2
        c_dut, _ = nnls(A / d_var[:, None], np.ones_like(d_var))
        dut_Ns.append(np.sqrt(c_dut[0]))
        dut_Ks.append(np.sqrt(c_dut[-1]))
        dut_tau_mins.append(tau[int(np.argmin(d_curve))] / 1.89)

    emp_N_std = float(np.std(dut_Ns, ddof=1))
    emp_K_std = float(np.std(dut_Ks, ddof=1))
    emp_tc_std = round(float(np.clip(np.std(dut_tau_mins, ddof=1), 50.0, 250.0)), 1)

    y_upper = y_mean_si + y_std_si

    def spread_loss(x_std):
        n_s, b_s, k_s = x_std
        if n_s < 0.0 or b_s < 0.0 or k_s < 0.0:
            return 1e9
        pred_up = compute_theoretical_upper_adev(
            tau, N_fit, B_single, K_fit, corr_time_single, n_s, b_s, k_s, B_poles_fit, Tc_poles_fit
        )
        rel = (pred_up - y_upper) / y_upper
        return float(np.mean(rel**2))

    init_std = [emp_N_std, 0.2 * B_single, max(emp_K_std, 0.1 * K_fit)]
    res_std = minimize(spread_loss, init_std, method="Nelder-Mead", options={"maxiter": 2000})
    N_std_fit, B_std_fit, K_std_fit = [float(max(0.0, v)) for v in res_std.x]

    return {
        "N": N_fit,
        "B": B_single,
        "K": K_fit,
        "correlation_time": corr_time_single,
        "B_poles": B_poles_fit,
        "correlation_time_poles": Tc_poles_fit,
        "N_std": N_std_fit,
        "B_std": B_std_fit,
        "K_std": K_std_fit,
        "correlation_time_std": emp_tc_std,
    }


def _format_sci_list(vals, precision=4):
    return "[" + ", ".join(f"{v:.{precision}e}" for v in vals) + "]"


def _format_float_list(vals, precision=2):
    return "[" + ", ".join(f"{v:.{precision}f}" for v in vals) + "]"


def update_yaml_section_in_place(yaml_text, section_name, fitted_axes):
    """
    Update a sensor section ('accelerometer' or 'gyroscope') in the YAML text
    while preserving all surrounding comments, spacing, and structure.
    """
    pattern = re.compile(
        rf"(^    {re.escape(section_name)}:\s*\n)(.*?)(?=^    [a-z_]+:|\Z|^  [a-z_]+:)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(yaml_text)
    if not match:
        raise ValueError(f"Could not find section '{section_name}:' in YAML file")

    header = match.group(1)
    body = match.group(2)

    N_list = [f["N"] for f in fitted_axes]
    B_list = [f["B"] for f in fitted_axes]
    K_list = [f["K"] for f in fitted_axes]
    tc_list = [f["correlation_time"] for f in fitted_axes]
    N_std_list = [f["N_std"] for f in fitted_axes]
    B_std_list = [f["B_std"] for f in fitted_axes]
    K_std_list = [f["K_std"] for f in fitted_axes]
    tc_std_list = [f["correlation_time_std"] for f in fitted_axes]

    def replace_scalar_array(text, key, formatted_arr):
        return re.sub(
            rf"(^      {re.escape(key)}:\s*)\[[^\]]*\]",
            rf"\g<1>{formatted_arr}",
            text,
            flags=re.MULTILINE,
        )

    body = replace_scalar_array(body, "N", _format_sci_list(N_list, 4))
    body = replace_scalar_array(body, "B", _format_sci_list(B_list, 4))
    body = replace_scalar_array(body, "K", _format_sci_list(K_list, 4))
    body = replace_scalar_array(body, "correlation_time", _format_float_list(tc_list, 1))
    body = replace_scalar_array(body, "N_std", _format_sci_list(N_std_list, 4))
    body = replace_scalar_array(body, "B_std", _format_sci_list(B_std_list, 4))
    if re.search(r"^      K_std:\s*\[", body, flags=re.MULTILINE):
        body = replace_scalar_array(body, "K_std", _format_sci_list(K_std_list, 4))
    body = replace_scalar_array(body, "correlation_time_std", _format_float_list(tc_std_list, 1))

    unit_comment = "m/s^2" if section_name == "accelerometer" else "rad/s"
    b_poles_lines = ["      B_poles:"]
    for idx, ax_name in enumerate(AXIS_NAMES):
        arr_str = _format_sci_list(fitted_axes[idx]["B_poles"], 4)
        b_poles_lines.append(f"        - {arr_str}  # {ax_name}-axis poles ({unit_comment})")
    b_poles_block = "\n".join(b_poles_lines)

    body = re.sub(
        r"^      B_poles:\s*\n(?:        - \[[^\]]*\][^\n]*\n){2}        - \[[^\]]*\][^\n]*",
        b_poles_block,
        body,
        flags=re.MULTILINE,
    )

    tc_poles_lines = ["      correlation_time_poles:"]
    for idx, ax_name in enumerate(AXIS_NAMES):
        arr_str = _format_float_list(fitted_axes[idx]["correlation_time_poles"], 2)
        padding = " " * max(2, 40 - len(f"        - {arr_str}"))
        tc_poles_lines.append(f"        - {arr_str}{padding}# {ax_name}-axis correlation times (s)")
    tc_poles_block = "\n".join(tc_poles_lines)

    body = re.sub(
        r"^      correlation_time_poles:\s*\n(?:        - \[[^\]]*\][^\n]*\n){2}        - \[[^\]]*\][^\n]*",
        tc_poles_block,
        body,
        flags=re.MULTILINE,
    )

    return yaml_text[: match.start()] + header + body + yaml_text[match.end() :]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_imudata = os.path.normpath(os.path.join(script_dir, "..", "imudata"))
    default_config = os.path.normpath(os.path.join(script_dir, "..", "config", "adis16607.yaml"))

    parser = argparse.ArgumentParser(
        description="Check hardware Allan deviation CSV data in imudata/ against YAML config and update if needed."
    )
    parser.add_argument(
        "--imudata-dir", default=default_imudata,
        help="Path to directory containing collected hardware CSVs (default: ../imudata)"
    )
    parser.add_argument(
        "--config", "-c", default=default_config,
        help="Path to simulator YAML config file (default: ../config/adis16607.yaml)"
    )
    parser.add_argument(
        "--tolerance-pct", type=float, default=1.0,
        help="Maximum allowed RMS relative error (%%) per axis before triggering YAML update (default: 1.0)"
    )
    parser.add_argument(
        "--max-tau", type=float, default=1e4,
        help="Maximum integration time tau (s) for fitting and checking (default: 1e4)"
    )
    parser.add_argument(
        "--poles", type=int, default=3, choices=[1, 2, 3, 4],
        help="Number of Gauss-Markov poles per axis when fitting (default: 3)"
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only check and report accuracy without modifying the YAML file"
    )
    parser.add_argument(
        "--force-update", action="store_true",
        help="Force re-fitting and updating the YAML file even if current parameters are within tolerance"
    )
    parser.add_argument(
        "--backup", action="store_true",
        help="Save a .bak copy of the YAML file before modifying it"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.imudata_dir):
        print(f"Error: imudata directory not found: {args.imudata_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"Error: YAML config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    print("=" * 78)
    print("IMU Simulator Hardware Data (imudata/) vs. YAML Configuration Check")
    print("=" * 78)
    print(f"  imudata directory : {args.imudata_dir}")
    print(f"  YAML config file  : {args.config}")
    print(f"  Tau range         : [0.5, {args.max_tau:g}] s")
    print(f"  RMS tolerance     : {args.tolerance_pct:.2f}%")

    xl_raw = load_imudata_sensor_group(
        args.imudata_dir, "plot_xl_avar.m", DEFAULT_XL_FILES_AND_EXCLUDED, "ug", "*XL*.csv"
    )
    gyro_raw = load_imudata_sensor_group(
        args.imudata_dir, "plot_gyro_avar.m", DEFAULT_GYRO_FILES_AND_EXCLUDED, "deg/h", "*Gyro*.csv"
    )

    with open(args.config, "r", encoding="utf-8") as f:
        yaml_text = f.read()
    config = yaml.safe_load(yaml_text)
    model_params = config["imu_simulator"]["model_parameter_settings"]

    sensor_specs = [
        ("accelerometer", "Accelerometer (ug)", xl_raw),
        ("gyroscope", "Gyroscope (deg/h)", gyro_raw),
    ]

    needs_update = False
    sensor_results = {}

    print("\n" + "-" * 78)
    print(f"{'Sensor / Axis':<20} | {'DUTs':>4} | {'RMS Err':>8} | {'Max Err':>8} | {'Spread Err':>10} | {'Convex':>6} | {'Status':<10}")
    print("-" * 78)

    for section_key, display_name, raw_data in sensor_specs:
        mask = (raw_data["tau"] >= 0.5) & (raw_data["tau"] <= args.max_tau)
        tau = raw_data["tau"][mask]
        mean_si = [m[mask] for m in raw_data["mean_adev_si"]]
        std_si = [s[mask] for s in raw_data["std_adev_si"]]
        dut_si = [d[mask, :] for d in raw_data["dut_adev_si"]]

        params = model_params[section_key]
        before_metrics = []
        section_needs_update = False

        for ax_idx, ax_name in enumerate(AXIS_NAMES):
            m = evaluate_axis_metrics(tau, mean_si[ax_idx], std_si[ax_idx], params, ax_idx)
            before_metrics.append(m)
            axis_ok = (m["rms_err_pct"] <= args.tolerance_pct) and m["is_convex"]
            if not axis_ok:
                needs_update = True
                section_needs_update = True
            status_str = "PASS" if axis_ok else "NEEDS FIT"
            convex_str = "YES" if m["is_convex"] else "HUMP!"
            label = f"{section_key[:5].upper()} {ax_name}-axis"
            print(
                f"{label:<20} | {raw_data['dut_count']:>4d} | "
                f"{m['rms_err_pct']:>7.2f}% | {m['max_err_pct']:>7.2f}% | "
                f"{m['upper_rms_err_pct']:>9.2f}% | {convex_str:>6} | {status_str:<10}"
            )

        sensor_results[section_key] = {
            "tau": tau,
            "mean_adev_si": mean_si,
            "std_adev_si": std_si,
            "dut_adev_si": dut_si,
            "dut_count": raw_data["dut_count"],
            "before_metrics": before_metrics,
            "section_needs_update": section_needs_update,
        }

    print("-" * 78)

    if args.check_only:
        if needs_update:
            print(
                f"\n[CHECK-ONLY] One or more axes exceeded the {args.tolerance_pct:.2f}% RMS tolerance "
                "or exhibited a non-convex hump. Run without --check-only to update the YAML."
            )
        else:
            print(
                f"\n[CHECK-ONLY] All axes match `imudata/` within {args.tolerance_pct:.2f}% RMS error "
                "and are strictly convex (hump-free)."
            )
        sys.exit(1 if needs_update else 0)

    if not needs_update and not args.force_update:
        print(
            f"\nAll axes already match `imudata/` within {args.tolerance_pct:.2f}% RMS tolerance "
            "(no YAML changes required; pass --force-update to re-fit anyway)."
        )
        sys.exit(0)

    print("\nFitting multi-pole Gauss-Markov bank & DUT spread parameters to `imudata/`...")
    updated_yaml_text = yaml_text

    for section_key, _, _ in sensor_specs:
        res = sensor_results[section_key]
        tau = res["tau"]
        mean_si = res["mean_adev_si"]
        std_si = res["std_adev_si"]
        dut_si = res["dut_adev_si"]

        fitted_axes = []
        for ax_idx, ax_name in enumerate(AXIS_NAMES):
            fit_res = fit_axis_multipole_and_spread(
                tau, mean_si[ax_idx], std_si[ax_idx], dut_si[ax_idx], n_poles=args.poles
            )
            fitted_axes.append(fit_res)

        updated_yaml_text = update_yaml_section_in_place(updated_yaml_text, section_key, fitted_axes)

    if args.backup:
        bak_path = args.config + ".bak"
        shutil.copy2(args.config, bak_path)
        print(f"Backed up previous YAML to: {bak_path}")

    with open(args.config, "w", encoding="utf-8") as f:
        f.write(updated_yaml_text)
    print(f"Updated YAML configuration in-place: {args.config}")

    new_config = yaml.safe_load(updated_yaml_text)
    new_model_params = new_config["imu_simulator"]["model_parameter_settings"]

    print("\nPost-Update Verification Table:")
    print("-" * 78)
    print(f"{'Sensor / Axis':<20} | {'DUTs':>4} | {'RMS Err':>8} | {'Max Err':>8} | {'Spread Err':>10} | {'Convex':>6} | {'Status':<10}")
    print("-" * 78)

    for section_key, _, _ in sensor_specs:
        res = sensor_results[section_key]
        tau = res["tau"]
        mean_si = res["mean_adev_si"]
        std_si = res["std_adev_si"]
        params = new_model_params[section_key]

        for ax_idx, ax_name in enumerate(AXIS_NAMES):
            m = evaluate_axis_metrics(tau, mean_si[ax_idx], std_si[ax_idx], params, ax_idx)
            axis_ok = (m["rms_err_pct"] <= args.tolerance_pct) and m["is_convex"]
            status_str = "PASS" if axis_ok else "WARN"
            convex_str = "YES" if m["is_convex"] else "HUMP!"
            label = f"{section_key[:5].upper()} {ax_name}-axis"
            print(
                f"{label:<20} | {res['dut_count']:>4d} | "
                f"{m['rms_err_pct']:>7.2f}% | {m['max_err_pct']:>7.2f}% | "
                f"{m['upper_rms_err_pct']:>9.2f}% | {convex_str:>6} | {status_str:<10}"
            )

    print("-" * 78)


if __name__ == "__main__":
    main()
