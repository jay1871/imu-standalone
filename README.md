# IMU Simulator Standalone

Standalone C++ IMU error model simulator extracted from [maximilian-nitsch/IMU-Simulator](https://github.com/maximilian-nitsch/IMU-Simulator). No ROS2 dependencies. Configured for the ADIS16607-2 with Allan variance validation.

Multi-backend architecture: C++ core (CPU), MuJoCo sensor plugin, JAX (CUDA/ROCm/TPU), NVIDIA Warp.

## Dependencies

### C++ Build (required)

| Package | Ubuntu/Debian | Purpose |
|---------|---------------|---------|
| CMake >= 3.14 | `cmake` | Build system |
| Eigen3 >= 3.3 | `libeigen3-dev` | Linear algebra (core library) |
| yaml-cpp | `libyaml-cpp-dev` | YAML config parsing |
| GTest | `libgtest-dev` | Unit tests |
| g++ (C++17) | `build-essential` | Compiler |

Install all:
```bash
sudo apt-get install -y build-essential cmake libeigen3-dev libyaml-cpp-dev libgtest-dev
```

### MuJoCo Plugin (optional — only if building the C sensor plugin)

```bash
# Install MuJoCo C library (needed for cmake -DIMU_BUILD_MUJOCO_PLUGIN=ON)
# Option A: from source
# Additional system dependencies for MuJoCo build (rendering/windowing):
sudo apt-get install -y libwayland-dev wayland-protocols libxkbcommon-dev \
    libx11-dev libxcursor-dev libxinerama-dev libxrandr-dev libxi-dev libgl1-mesa-dev

git clone https://github.com/google-deepmind/mujoco.git
cd mujoco && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/.local
cmake --build . -j$(nproc) && cmake --install .

# Option B: download prebuilt release <--preferred method
# https://github.com/google-deepmind/mujoco/releases

# Install MuJoCo Python bindings (needed for plugin integration tests and usage)
pip install mujoco
```

### Python Environment Setup

You can use either Python's built-in `venv` (recommended on Ubuntu/Debian) or `conda`. Python 3.10 through 3.12 is recommended.

#### Option A: Standard Python `venv` (Recommended)

```bash
# Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

#### Option B: Conda Environment

```bash
# Create and activate conda environment
conda create -n imu python=3.11 -y
conda activate imu
```

> **Important — MuJoCo Python version must match the C plugin build.**
> The plugin `.so` is compiled against the MuJoCo C library version set by `MUJOCO_ROOT`.
> The Python `mujoco` package must be the **same version**, or the plugin will fail
> to load with `"plugin not found"`.
> ```bash
> # Match your MUJOCO_ROOT build (e.g. mujoco-3.11.0):
> pip install mujoco==3.11.0
> ```

### Installing Python Dependencies & Package

Activate your environment, then install dependencies:

```bash
# Install core analysis & testing dependencies
pip install -r scripts/requirements.txt

# Install the imu_nbk package in editable mode:
cd python
pip install -e .                 # Base NumPy backend
pip install -e ".[test]"          # + Test runner and dependencies

# Optional GPU acceleration backends:
pip install -e ".[jax-cpu]"       # JAX CPU only
pip install -e ".[jax-cuda]"      # JAX with NVIDIA CUDA 12
pip install -e ".[warp]"          # NVIDIA Warp (requires CUDA)
```

---

## Build

### C++ Core (no MuJoCo)

```bash
cd imu-standalone
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### C++ with MuJoCo Plugin

```bash
cd imu-standalone/build

# Point MUJOCO_ROOT to your prebuilt MuJoCo release directory:
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DIMU_BUILD_MUJOCO_PLUGIN=ON \
    -DMUJOCO_ROOT=/path/to/mujoco-3.x.x
cmake --build . -j$(nproc)

# Output: mujoco_plugin/imu_nbk_plugin.so
```

> **Note:** If your path contains spaces, ensure the path passed to `MUJOCO_ROOT` is properly quoted or symlinked to a space-free directory.

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | ON | Build C++ unit tests |
| `BUILD_APPS` | ON | Build allan_validation executable |
| `BUILD_SHARED_LIBS` | OFF | Static libs (recommended for plugin linking) |
| `IMU_BUILD_MUJOCO_PLUGIN` | OFF | Build MuJoCo C sensor plugin |

### Python Package (imu_nbk)

```bash
cd imu-standalone/python

# Base install (NumPy backend only — requires numpy, pyyaml)
pip install -e .

# With JAX on NVIDIA CUDA (requires: pip install "jax[cuda12]")
pip install -e ".[jax-cuda]"

# With JAX on AMD ROCm (requires: pip install jax-rocm7)
pip install -e ".[jax-rocm]"

# With JAX CPU-only (no GPU needed, for testing)
pip install -e ".[jax-cpu]"

# With NVIDIA Warp (requires: pip install warp-lang, NVIDIA GPU)
pip install -e ".[warp]"

# With test dependencies (pytest, allantools, matplotlib)
pip install -e ".[test]"

# Full install on NVIDIA machine
pip install -e ".[jax-cuda,warp,test]"
```

---

## Testing

### C++ Unit Tests

```bash
cd imu-standalone/build
ctest --output-on-failure
```

### Python Tests

```bash
cd imu-standalone/python

# Fast tests: deterministic cross-backend math verification & MuJoCo plugin integration
pytest tests/test_deterministic.py tests/test_mujoco_plugin.py -v

# Run all non-slow tests:
pytest -m "not slow" -v

# Statistical Allan variance equivalence (slow, ~2-3 minutes):
pytest tests/test_allan_equiv.py -v
```

### Golden ADEV Regression

Generate the golden reference (run once, store for future comparisons):
```bash
cd imu-standalone/build
./allan_validation ../config/adis16607.yaml ../test/golden/golden_adev_1hr.csv 1 1000
```

Any C++ code change must reproduce this file bitwise (same seed = deterministic output).
The output is a compact ADEV results file (~2 KB), not the full time series.

---

## Simulation

The IMU error model integrates with MuJoCo physics in two ways:

| Mode | Backend | Hardware | Use Case |
|------|---------|----------|----------|
| `plugin` | MuJoCo C plugin | CPU | Single-env validation, digital twin, HIL |
| `mjx` | MJX physics + JAX noise | GPU (CUDA/ROCm/TPU) | Batched RL training (1000s of envs) |
| `jax-only` | JAX noise, no physics | GPU or CPU | Allan variance, noise characterization |

### Quick Start

```bash
cd imu-standalone/scripts

# Free-falling body — MuJoCo C plugin (requires plugin built)
python simulate_imu.py --mode plugin --model ../models/free_body_imu.xml --duration 5

# Spinning body with motor torque
python simulate_imu.py --mode plugin --model ../models/spinning_body_imu.xml --ctrl 0.5 --duration 10

# MJX on GPU — 16 parallel environments
python simulate_imu.py --mode mjx --model ../models/free_body_imu.xml --n-envs 16 --duration 5

# Pure noise only (no MuJoCo needed — just JAX)
python simulate_imu.py --mode jax-only --duration 60

# Save CSV without plotting
python simulate_imu.py --mode plugin --model ../models/free_body_imu.xml --no-plot -o data.csv
```

### MuJoCo XML Models

Models live in `models/` and define the physical scene. The XML tells MuJoCo:
- What bodies exist, their mass/inertia, joints, contacts
- Where the IMU is mounted (a `<site>` element)
- Which sensors to compute (noiseless reference + noisy plugin)

**Included models:**

| Model | What it does |
|-------|-------------|
| `free_body_imu.xml` | Box in free-fall. Accelerometer sees gravity, gyro reads zero. Simplest validation. |
| `spinning_body_imu.xml` | Arm on a hinge with motor. Tests gyro + centripetal acceleration coupling. |

**To create your own:** Copy `free_body_imu.xml`, replace the `<worldbody>` with your robot, and attach `<site name="imu_site">` where your IMU is physically mounted. Keep the `<extension>` and `<sensor>` blocks unchanged. **Important:** The built-in `accelerometer` and `gyro` sensors must be defined *before* the plugin sensor in the XML — the plugin reads their output, so they must be computed first.

### How the Plugin Mode Works

```
┌─────────────────────────────────────────────────────────────────┐
│  MuJoCo steps physics (forces, contacts, gravity)               │
│      ↓                                                          │
│  Built-in accelerometer/gyro sensors compute true specific      │
│  force and angular velocity at the IMU site (body frame)        │
│      ↓                                                          │
│  Plugin reads these correct values from d->sensordata           │
│  (requires acc_true/gyro_true defined BEFORE plugin in XML)     │
│      ↓                                                          │
│  Plugin passes through ImuSimulator (full NBK error model):     │
│    - Applies deterministic errors (bias, scale factor, misalign)│
│    - Applies stochastic errors (N, B, K via Gauss-Markov)       │
│      ↓                                                          │
│  Output: d->sensordata[offset:offset+6] = noisy [acc, gyro]    │
└─────────────────────────────────────────────────────────────────┘
```

### How the MJX Mode Works (GPU Batched)

```
┌─────────────────────────────────────────────────────────────────┐
│  MJX steps physics on GPU (all N_ENVS in parallel via vmap)     │
│      ↓                                                          │
│  Built-in accelerometer/gyro sensors give noiseless readings    │
│  (MJX has no plugin API — these are MuJoCo's native sensors)   │
│      ↓                                                          │
│  imu_nbk.jax_backend.batched_step() adds NBK noise on GPU      │
│  (same math as C++ plugin, JIT-compiled, vmapped over envs)     │
│      ↓                                                          │
│  obs_acc = true_acc + noise[:, :3]                              │
│  obs_gyro = true_gyro + noise[:, 3:]                            │
│      ↓                                                          │
│  Feed to RL policy — trains robust to realistic sensor noise    │
└─────────────────────────────────────────────────────────────────┘
```

### RL Training Example

See `scripts/mjx_training_example.py` for a complete 1024-environment training loop pattern.

### Viewing Results

Three ways to see simulation output:

#### 1. Interactive 3D Viewer (real-time)

Watch the body move in MuJoCo's 3D viewer while sensor data prints to terminal:

```bash
cd imu-standalone/scripts

# Free-falling body
python view_simulation.py ../models/free_body_imu.xml

# Spinning body with motor torque
python view_simulation.py ../models/spinning_body_imu.xml --ctrl 0.3
```

Requires a display (X11/Wayland). Prints noisy vs true accelerometer values every 100 steps.

**Keyboard controls:**

| Key | Action |
|-----|--------|
| SPACE | Pause / resume simulation |
| UP or `]` | Speed up 2x (max 64x) |
| DOWN or `[` | Slow down 2x (min 1/16x) |
| R | Reset simulation to initial state |
| Q or ESC | Quit |
| Ctrl+C | Force quit (terminal) |

**Mouse controls (MuJoCo viewer built-in):**

| Action | Effect |
|--------|--------|
| Left-drag | Rotate camera |
| Right-drag | Pan camera |
| Scroll | Zoom in/out |
| Double-click body | Track that body |

#### 2. Run Simulation → CSV → Post-hoc Plotting

```bash
cd imu-standalone/scripts

# Step 1: Generate data (saves CSV)
python simulate_imu.py --mode plugin --model ../models/free_body_imu.xml --duration 10 --no-plot -o free_fall.csv

# Step 2: Analyze with plot_results.py
python plot_results.py free_fall.csv                        # true vs noisy (6 panels)
python plot_results.py free_fall.csv --noise-only           # just the error component
python plot_results.py free_fall.csv --psd                  # power spectral density
python plot_results.py free_fall.csv --tstart 2 --tend 5    # zoom to time window
python plot_results.py free_fall.csv --stats                # print mean/std per axis
python plot_results.py free_fall.csv --save output.png --no-show  # headless (no display)

# Compare two simulation modes side-by-side
python plot_results.py imu_sim_plugin.csv imu_sim_mjx.csv
```

#### 3. Auto-generated Plots (one-shot)

`simulate_imu.py` generates plots automatically unless `--no-plot` is passed:

```bash
python simulate_imu.py --mode plugin --model ../models/free_body_imu.xml --duration 5
# Outputs: imu_sim_plugin.csv + imu_simulation.png + imu_noise.png
```

#### 4. Long-Duration Drift Analysis (5–10 hours)

Shows how bias instability and random walk accumulate over hours with a stationary IMU:

```bash
cd imu-standalone/scripts

# 10-hour run (default) — uses JAX if available, falls back to NumPy
python drift_analysis.py

# 5-hour run at 100 Hz (faster computation, same statistics)
python drift_analysis.py --duration 5 --rate 100

# Force NumPy backend (slower but no JAX dependency)
python drift_analysis.py --duration 1 --backend numpy

# Just the summary plot (attitude + position drift)
python drift_analysis.py --duration 10 --summary-only

# Headless (save CSV, no plots)
python drift_analysis.py --duration 10 --save drift_10hr.csv --no-plot
```

**What it produces:**

| Plot | Shows |
|------|-------|
| `drift_analysis.png` | 5-row × 3-col: raw noise, attitude drift (deg), position drift (m), internal z_B and z_K states |
| `drift_summary.png` | 2-panel: attitude drift + position drift with final values labeled |
| `drift_10hr.csv` | Decimated time series (all states, integrals) for post-hoc analysis |

**What you'll observe:**
- **Attitude drift**: gyro bias instability (~2 deg/hr on Y axis) causes degrees of heading error
- **Position drift**: accelerometer errors integrated twice → meters of position error
- **z_B (Gauss-Markov)**: oscillates with correlation time (200–700s), bounded
- **z_K (random walk)**: grows unbounded as √t — this is what drives long-term drift

#### Output Scripts Summary

| Script | What it produces |
|--------|-----------------|
| `view_simulation.py` | Live 3D viewer + terminal sensor printout |
| `simulate_imu.py` | CSV data file + auto-generated plots (PNG) |
| `plot_results.py` | Post-hoc analysis: true vs noisy, noise-only, PSD, multi-file comparison |
| `drift_analysis.py` | Long-duration drift/bias visualization (hours of stationary IMU) |
| `mjx_training_example.py` | Demonstrates 1024-env GPU RL training loop (prints shapes/values) |

---

## Usage

### Allan Variance Validation

The `allan_validation` tool runs a stationary IMU simulation and computes overlapping Allan
deviation **in-memory** — no multi-GB CSV files. It outputs a compact ADEV results file
(~2 KB) with one row per tau point.

```bash
cd imu-standalone/build

# Single seed, 10 hours at 1 kHz (~30s on CPU, ~1.7 GB RAM during run)
./allan_validation ../config/adis16607.yaml adev.csv 10 1000

# Plot results with datasheet reference markers
python3 ../scripts/allan_variance.py adev.csv ../config/adis16607.yaml --rate 1000
```

**Multi-seed** (mean ± σ envelope, matches datasheet methodology):

```bash
# 100 seeds, 10 hours each (~50 min total on CPU)
./allan_validation ../config/adis16607.yaml adev_100seeds.csv 10 1000 100

# Plot shows shaded ±σ bands + datasheet markers
python3 ../scripts/allan_variance.py adev_100seeds.csv ../config/adis16607.yaml --rate 1000
```

**Arguments:** `<config.yaml> [output.csv] [duration_hours] [sample_rate_hz] [num_seeds]`

| Argument | Default | Description |
|----------|---------|-------------|
| config.yaml | (required) | IMU parameter config |
| output.csv | `allan_adev.csv` | Output ADEV results |
| duration_hours | 10 | Simulation duration per seed |
| sample_rate_hz | config value | Overrides config `sample_time` |
| num_seeds | 1 | Number of independent runs (seeds: config.seed, +1, +2, ...) |

**Notes:**
- All computation runs on CPU (C++, Eigen3, std::mt19937). No GPU required.
- Multi-seed runs use sequential deterministic seeds (config.seed + 0, +1, +2, ...) for reproducibility.
- The first seed in a multi-seed run matches a single-seed run — results are reproducible.
- RAM usage: ~1.7 GB per 10-hour run at 1 kHz (time series held in memory during ADEV computation, then freed).
- To resolve the gyro bias instability minimum (corrTime = 1500–2000s), runs of ≥ 5 hours are recommended.

**Output format:**

Single seed:
```
tau,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
```

Multi-seed:
```
tau,acc_x_mean,acc_x_std,acc_y_mean,acc_y_std,...,gyro_z_mean,gyro_z_std
```

### MuJoCo Plugin (C/C++)

XML model:
```xml
<mujoco>
  <extension>
    <plugin plugin="mujoco.sensor.imu_nbk"/>
  </extension>
  <worldbody>
    <body name="robot_base">
      <site name="imu_site" pos="0 0 0.1"/>
    </body>
  </worldbody>
  <!-- Noiseless reference sensors — MUST come before the plugin sensor -->
  <sensor>
    <accelerometer name="acc_true" site="imu_site"/>
    <gyro name="gyro_true" site="imu_site"/>
  </sensor>
  <!-- NBK error model plugin (reads from acc_true/gyro_true above) -->
  <sensor>
    <plugin name="imu_noisy" plugin="mujoco.sensor.imu_nbk"
            objtype="site" objname="imu_site">
      <config key="config_path" value="config/adis16607.yaml"/>
    </plugin>
  </sensor>
</mujoco>
```

Python usage:
```python
import mujoco

mujoco.mj_loadPluginLibrary("build/mujoco_plugin/imu_nbk_plugin.so")
model = mujoco.MjModel.from_xml_path("my_robot.xml")
data = mujoco.MjData(model)

mujoco.mj_step(model, data)
imu_reading = data.sensordata[:6]  # [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
```

### JAX Backend (batched GPU training)

```python
import jax
from imu_nbk import load_config
from imu_nbk.jax_backend import init_state_batched, batched_step, params_to_jax

params = load_config("config/adis16607.yaml")
jax_params = params_to_jax(params)

n_envs = 1024
states = init_state_batched(n_envs)
rng_keys = jax.random.split(jax.random.PRNGKey(0), n_envs)

# In training loop (after mjx.step):
states, noises, rng_keys = batched_step(states, jax_params, rng_keys)
# noises shape: (1024, 6) = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z] per env
```

### NVIDIA Warp Backend (CUDA only)

```python
from imu_nbk import load_config
from imu_nbk.warp_backend import ImuNbkWarp

params = load_config("config/adis16607.yaml")
sim = ImuNbkWarp(params, n_envs=1024, seed=42)

output = sim.step()           # wp.array shape (1024, 6) on GPU
numpy_out = sim.get_output_numpy()  # copy to host
```

### NumPy Backend (reference/testing)

```python
import numpy as np
from imu_nbk import load_config, get_backend

params = load_config("config/adis16607.yaml")
backend = get_backend("numpy")

state = backend.init_state()
rng = np.random.default_rng(42)

for _ in range(n_steps):
    state, noise = backend.step(state, params, rng)
    # noise shape (6,): [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
```

---

## Project Structure

```
imu-standalone/
├── include/imu_simulator/
│   ├── imu_simulator.h              # Core simulator class (Eigen3 only)
│   ├── imu_simulator_structures.h   # Data structures
│   └── config_loader.h              # YAML config loader
├── src/
│   ├── imu_simulator.cpp            # Core implementation
│   └── config_loader.cpp            # YAML parsing
├── models/
│   ├── free_body_imu.xml            # Free-fall body (validation)
│   └── spinning_body_imu.xml        # Hinge + motor (gyro test)
├── mujoco_plugin/
│   ├── CMakeLists.txt               # Plugin build (requires mujoco)
│   ├── imu_nbk_plugin.h             # Plugin class
│   ├── imu_nbk_plugin.cpp           # Compute/reset/init callbacks
│   └── register.cpp                 # mjPLUGIN_LIB_INIT entry point
├── python/
│   ├── pyproject.toml               # Python package config
│   ├── imu_nbk/
│   │   ├── __init__.py              # Backend selector
│   │   ├── params.py                # YAML → ImuNbkParams dataclass
│   │   ├── numpy_backend.py         # Reference implementation
│   │   ├── jax_backend.py           # JAX/MJX batched (CUDA/ROCm/TPU)
│   │   └── warp_backend.py          # NVIDIA Warp kernel (CUDA only)
│   └── tests/
│       ├── test_deterministic.py    # Fixed-draw cross-backend match
│       ├── test_allan_equiv.py      # Statistical ADEV verification
│       └── test_mujoco_plugin.py    # Plugin integration
├── apps/
│   └── allan_validation.cpp         # Static sim → CSV
├── config/
│   └── adis16607.yaml               # ADIS16607-2 parameters (SI units)
├── scripts/
│   ├── simulate_imu.py              # Unified simulation (plugin/mjx/jax-only)
│   ├── view_simulation.py           # Interactive 3D viewer + live sensor output
│   ├── plot_results.py              # Post-hoc CSV analysis and plotting
│   ├── drift_analysis.py            # Long-duration drift/bias visualization
│   ├── mjx_training_example.py      # RL training loop pattern (1024 envs)
│   ├── allan_variance.py            # Allan deviation analysis + plotting
│   ├── requirements.txt             # Python dependencies (analysis only)
│   └── setup_and_build.sh           # Build helper
├── test/
│   ├── imu_simulator_test.cpp       # C++ unit tests
│   └── golden/                      # Regression reference CSV
└── CMakeLists.txt
```

## Library Architecture

Three-tier design:

- **`imu_simulator`** (C++, Eigen3 only) — core simulation math. Link from MuJoCo plugins or Isaac Sim.
- **`imu_simulator_config`** (C++, adds yaml-cpp) — YAML config loader. Use for standalone tools.
- **`imu_nbk`** (Python) — GPU backends (JAX/Warp) + NumPy reference. Runtime backend selection.

All read the same `config/adis16607.yaml`. No math duplication — every backend implements the identical NBK discrete-time state-space model.

## ADIS16607 Configuration

Parameters in `config/adis16607.yaml` are converted to SI units from the ADIS16607 datasheet Table 1. Parameters not available from the datasheet are marked with `ESTIMATE` comments — update these after running Allan variance characterization on real hardware:

```bash
grep "ESTIMATE" config/adis16607.yaml
```

These are:
- **K** (rate/acceleration random walk) — determines long-tau slope (+1/2)
- **correlation_time** — Gauss-Markov time constant, affects bias instability plateau location

### Random Seed

The `seed` field in `general_settings` controls RNG behavior:

| Value | Behavior |
|-------|----------|
| `seed: 42` (or any positive integer) | Deterministic — same seed produces identical output every run. Reset also reproduces the same sequence. |
| `seed: -1` | Truly random — seeded from hardware entropy (`std::random_device`). Each run and reset produces different noise. |

Use a positive seed for repeatable experiments and regression testing. Use `-1` for Monte Carlo runs or when you want independent realizations.

### Gravity Model (`enable_local_ref_vec`)

The `ImuSimulator` has an internal gravity model (`enableLocalRefVec`) that projects gravity into the body frame and subtracts it from the input acceleration to produce specific force. Whether this should be enabled depends on what the caller passes in:

| Context | Gravity model | Reason |
|---------|:---:|--------|
| **MuJoCo plugin** | OFF | MuJoCo's built-in accelerometer already outputs specific force (includes gravity). Plugin disables the model to avoid double-subtraction. |
| **Allan validation** | OFF | Pure noise characterization — zero input, only stochastic errors. A gravity offset would create a large DC ramp in the integrated phase data, destroying floating-point precision at long tau. |
| **Standalone (navigation)** | ON (config default) | Caller passes coordinate acceleration (without gravity). The model projects gravity via the body-to-nav quaternion to produce specific force. This is the classic INS error model usage. |
| **MJX / JAX backend** | N/A | Noise is applied as a post-step addition to MJX's noiseless sensor output. No gravity model involved — MJX sensors already include gravity. |

The YAML config sets `enable_local_ref_vec: true` by default (standalone navigation use case). The MuJoCo plugin and Allan validation tool override this programmatically.

## References

- Nitsch, M. et al. — [IMU-Simulator](https://github.com/maximilian-nitsch/IMU-Simulator)
- Farrell, J. et al. — "Inertial Measurement Unit Error Modeling Tutorial", IEEE Control Systems Magazine, 2022
- ADIS16607 Datasheet, Analog Devices
- [MuJoCo Plugin API](https://mujoco.readthedocs.io/en/stable/programming/extension.html)
- [MJX Documentation](https://mujoco.readthedocs.io/en/stable/mjx.html)
