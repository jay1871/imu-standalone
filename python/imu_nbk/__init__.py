"""Multi-backend NBK stochastic IMU error model.

Backends:
  numpy - Reference implementation (always available)
  jax   - GPU-accelerated via JAX (CUDA/ROCm/TPU)
  warp  - NVIDIA Warp kernel (CUDA only)
"""

import importlib


def get_backend(name="numpy"):
    """Import and return the requested backend module.

    Args:
        name: One of "numpy", "jax", "warp"

    Returns:
        Module with step() function and init_state() factory.
    """
    if name == "jax":
        return importlib.import_module(".jax_backend", package="imu_nbk")
    elif name == "warp":
        return importlib.import_module(".warp_backend", package="imu_nbk")
    elif name == "numpy":
        return importlib.import_module(".numpy_backend", package="imu_nbk")
    else:
        raise ValueError(f"Unknown backend: {name!r}. Use 'numpy', 'jax', or 'warp'.")


from .params import ImuNbkParams, load_config  # noqa: E402, F401
