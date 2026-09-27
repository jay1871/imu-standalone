"""Load IMU NBK parameters from the shared YAML config.

Reads the same config/adis16607.yaml used by the C++ ImuSimulator,
producing a frozen dataclass with all parameters in SI units.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class ImuNbkParams:
    """NBK stochastic model parameters for both accel and gyro (per-axis)."""

    # Accelerometer stochastic params — shape (3,)
    acc_N: np.ndarray
    acc_B: np.ndarray
    acc_K: np.ndarray
    acc_corr_time: np.ndarray

    # Gyroscope stochastic params — shape (3,)
    gyro_N: np.ndarray
    gyro_B: np.ndarray
    gyro_K: np.ndarray
    gyro_corr_time: np.ndarray

    # Deterministic error intervals
    acc_turn_on_bias_interval: np.ndarray
    acc_meas_range: np.ndarray
    acc_scale_factor_interval: np.ndarray
    acc_misalignment_interval: np.ndarray
    acc_resolution: np.ndarray

    gyro_turn_on_bias_interval: np.ndarray
    gyro_meas_range: np.ndarray
    gyro_scale_factor_interval: np.ndarray
    gyro_misalignment_interval: np.ndarray
    gyro_resolution: np.ndarray

    # Timing
    dt: float

    def __eq__(self, other):
        if not isinstance(other, ImuNbkParams):
            return False
        for field in self.__dataclass_fields__:
            a = getattr(self, field)
            b = getattr(other, field)
            if isinstance(a, np.ndarray):
                if not np.array_equal(a, b):
                    return False
            else:
                if a != b:
                    return False
        return True


def load_config(yaml_path: str | Path) -> ImuNbkParams:
    """Load parameters from the shared YAML config file.

    Args:
        yaml_path: Path to adis16607.yaml (or compatible YAML).

    Returns:
        ImuNbkParams with all values in SI units.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg["imu_simulator"]
    acc = sim_cfg["model_parameter_settings"]["accelerometer"]
    gyro = sim_cfg["model_parameter_settings"]["gyroscope"]
    dt = sim_cfg["general_settings"]["sample_time"]

    return ImuNbkParams(
        acc_N=np.array(acc["N"], dtype=np.float64),
        acc_B=np.array(acc["B"], dtype=np.float64),
        acc_K=np.array(acc["K"], dtype=np.float64),
        acc_corr_time=np.array(acc["correlation_time"], dtype=np.float64),
        acc_turn_on_bias_interval=np.array(acc["interval_turn_on_bias"], dtype=np.float64),
        acc_meas_range=np.array(acc["measurement_range"], dtype=np.float64),
        acc_scale_factor_interval=np.array(acc["interval_scale_factor"], dtype=np.float64),
        acc_misalignment_interval=np.array(acc["interval_misalignment"], dtype=np.float64),
        acc_resolution=np.array(acc["resolution"], dtype=np.float64),
        gyro_N=np.array(gyro["N"], dtype=np.float64),
        gyro_B=np.array(gyro["B"], dtype=np.float64),
        gyro_K=np.array(gyro["K"], dtype=np.float64),
        gyro_corr_time=np.array(gyro["correlation_time"], dtype=np.float64),
        gyro_turn_on_bias_interval=np.array(gyro["interval_turn_on_bias"], dtype=np.float64),
        gyro_meas_range=np.array(gyro["measurement_range"], dtype=np.float64),
        gyro_scale_factor_interval=np.array(gyro["interval_scale_factor"], dtype=np.float64),
        gyro_misalignment_interval=np.array(gyro["interval_misalignment"], dtype=np.float64),
        gyro_resolution=np.array(gyro["resolution"], dtype=np.float64),
        dt=float(dt),
    )
