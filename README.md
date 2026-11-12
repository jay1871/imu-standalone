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

# Output: mujoco_plugin/imu_nbk_plugin.so, allan_validation, imu_simulator_test
```

> **Note:** `ENABLE_DUT_PARAMETER_SPREAD` is `ON` by default in `CMakeLists.txt`. This does **not** affect the MuJoCo plugin (`imu_nbk_plugin.so`) or single-DUT simulation platform runs (which always use the nominal 1-DUT parameters); it only activates automatically inside `allan_validation` when running multi-seed population characterization (`num_seeds > 1`). If your path contains spaces, ensure the path passed to `MUJOCO_ROOT` is properly quoted or symlinked to a space-free directory.

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | ON | Build C++ unit tests |
| `BUILD_APPS` | ON | Build `allan_validation` executable (with OpenMP multi-seed parallelization when available) |
| `BUILD_SHARED_LIBS` | OFF | Static libs (recommended for plugin linking) |
| `IMU_BUILD_MUJOCO_PLUGIN` | OFF | Build MuJoCo C sensor plugin |
| `ENABLE_DUT_PARAMETER_SPREAD` | ON | Compile support for multi-DUT part-to-part parameter spread (`N_std`, `B_std`, `K_std`, `correlation_time_std`) used automatically by `allan_validation` when `num_seeds > 1` |

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
| `allan_variance.py` | Plots simulated Allan deviation curves (`sim mean` and `sim mean + σ`) |
| `calibrate_yaml_from_imudata.py` | Checks collected hardware CSVs in `imudata/` against `config/adis16607.yaml` and updates the YAML in-place |
| `mjx_training_example.py` | Demonstrates 1024-env GPU RL training loop (prints shapes/values) |

---

## Usage

### Allan Variance Validation & Plotting

The `allan_validation` tool runs a stationary IMU simulation and computes overlapping Allan
deviation **in-memory** — no multi-GB CSV files. Multi-seed runs are parallelized with **OpenMP**
and automatically cap the number of concurrent threads based on available system RAM (~3.3 GB per thread
for a 10-hour 1000 Hz run). It outputs a compact ADEV results file (~2–5 KB) with one row per τ point.

#### Situation 1: Single-Seed Run (Single Nominal DUT)

When `num_seeds = 1`, `allan_validation` automatically runs with `DUT Spread: DISABLED (single nominal DUT)` to validate a single nominal sensor:

```bash
cd imu-standalone/build

# Single seed (1 nominal DUT), 10 hours at 1 kHz
./allan_validation ../config/adis16607.yaml adev_1seed.csv 10 1000 1

# Plot simulated Allan deviation (2-panel: Accelerometer + Gyroscope)
python3 ../scripts/allan_variance.py adev_1seed.csv ../config/adis16607.yaml \
    --output ../scripts/allan_validation_1seed_plot.png
```

#### Situation 2: Multi-DUT Population Validation (33, 100, or 150 Seeds — Automatic Spread)

When `num_seeds > 1`, `allan_validation` **automatically enables** `DUT Spread: ENABLED (multi-DUT population)`, sampling each virtual DUT from `N_std`, `B_std`, `K_std`, and `correlation_time_std` in `config/adis16607.yaml` (and scaling `B_poles` / `correlation_time_poles` proportionally) to validate both `sim mean` and `sim mean + σ` across a population of sensors:

```bash
cd imu-standalone/build

# 33 virtual DUTs, 10 hours per seed at 1000 Hz
./allan_validation ../config/adis16607.yaml adev_10hr_spread_33seeds.csv 10 1000 33

# 100 virtual DUTs, 10 hours per seed at 1000 Hz
./allan_validation ../config/adis16607.yaml adev_10hr_spread_100seeds.csv 10 1000 100

# 150 virtual DUTs, 10 hours per seed at 1000 Hz
./allan_validation ../config/adis16607.yaml adev_10hr_spread_150seeds.csv 10 1000 150
```

#### Situation 3: Multi-Seed Run of a Single Nominal DUT (`--no-spread`)

If you want to run multiple random seeds of the **exact same nominal sensor** without part-to-part DUT variation, pass `--no-spread`:

```bash
cd imu-standalone/build

# 33 seeds of 1 nominal DUT (DUT Spread: DISABLED)
./allan_validation ../config/adis16607.yaml adev_10hr_multipole_33seeds.csv 10 1000 33 --no-spread

# 100 seeds of 1 nominal DUT (DUT Spread: DISABLED)
./allan_validation ../config/adis16607.yaml adev_10hr_multipole_100seeds.csv 10 1000 100 --no-spread
```

#### Generating Allan Deviation Plots (`scripts/allan_variance.py`)

`scripts/allan_variance.py` plots **only the simulation data** (`sim mean` and `sim mean + σ`):

```bash
# Combined 2-panel plots (Accelerometer + Gyroscope, simulation data only)
python3 ../scripts/allan_variance.py adev_10hr_spread_33seeds.csv ../config/adis16607.yaml \
    --output ../scripts/allan_validation_spread_33seeds_plot.png

python3 ../scripts/allan_variance.py adev_10hr_spread_100seeds.csv ../config/adis16607.yaml \
    --output ../scripts/allan_validation_spread_100seeds_plot.png

python3 ../scripts/allan_variance.py adev_10hr_spread_150seeds.csv ../config/adis16607.yaml \
    --output ../scripts/allan_validation_spread_150seeds_plot.png

# Single-panel plots (Accelerometer-only or Gyroscope-only)
python3 ../scripts/allan_variance.py adev_10hr_spread_100seeds.csv ../config/adis16607.yaml \
    --xl-only --output xl_adev.png
python3 ../scripts/allan_variance.py adev_10hr_spread_100seeds.csv ../config/adis16607.yaml \
    --gyro-only --output gyro_adev.png

# Optional: overlay theoretical YAML model curve on top of simulation data
python3 ../scripts/allan_variance.py adev_10hr_spread_100seeds.csv ../config/adis16607.yaml \
    --show-theory --output adev_with_theory.png
```

**Arguments (`allan_validation`):** `<config.yaml> [output.csv] [duration_hours] [sample_rate_hz] [num_seeds] [--no-spread]`

| Argument | Default | Description |
|----------|---------|-------------|
| `config.yaml` | (required) | IMU parameter config |
| `output.csv` | `allan_adev.csv` | Output ADEV results |
| `duration_hours` | `10` | Simulation duration per seed (`10` hours covers τ in `[0.5, 10000] s`) |
| `sample_rate_hz` | config value | Overrides config `sample_time` |
| `num_seeds` | `1` | Number of independent runs (seeds: `config.seed + 0, +1, +2, ...`) |
| `--no-spread` | (optional) | Disable multi-DUT parameter spread during a multi-seed run (simulates 1 nominal DUT across all seeds) |

**Automatic Multi-DUT Parameter Spread (`ENABLE_DUT_PARAMETER_SPREAD=ON` by default):**
- When `num_seeds == 1` (single-seed run), `allan_validation` always simulates the exact nominal 1-DUT parameters (`N`, `B`, `K`, `correlation_time`, `B_poles`, `correlation_time_poles`).
- When `num_seeds > 1` (multi-seed run, e.g. 33, 100, or 150 seeds), `allan_validation` **automatically enables** multi-DUT parameter spread (`DUT Spread: ENABLED`), sampling each virtual DUT from `N_std`, `B_std`, `K_std`, and `correlation_time_std` in `config/adis16607.yaml` (and scaling `B_poles` / `correlation_time_poles` proportionally) to reproduce both the mean and +1σ spread of a physical sensor population.
- Pass `--no-spread` on the command line if you ever want a multi-seed run of a single nominal DUT without part-to-part variation.

**Notes:**
- All computation runs on CPU (C++, Eigen3, `std::mt19937`, OpenMP) using 80-bit `long double` phase integration to prevent floating-point precision loss over multi-hour runs. No GPU required.
- Multi-seed runs accumulate results in deterministic seed order (`0, 1, ..., num_seeds - 1`) via Welford's algorithm, so results are bitwise identical regardless of the number of OpenMP threads used.
- The first seed in a nominal multi-seed run matches a single-seed run.

**Output format:**

Single seed:
```
tau,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
```

Multi-seed:
```
tau,acc_x_mean,acc_x_std,acc_y_mean,acc_y_std,...,gyro_z_mean,gyro_z_std
```

---

### Updating the YAML Config with Hardware Data (`imudata/`)

Collected hardware Allan deviation CSV files and MATLAB reference scripts (`plot_gyro_avar.m`, `plot_xl_avar.m`) live in the `imudata/` directory. The script `scripts/calibrate_yaml_from_imudata.py` checks the hardware measurements in `imudata/` against `config/adis16607.yaml` and automatically updates the YAML parameters in-place (preserving all comments and formatting) when needed.

#### How to Use `calibrate_yaml_from_imudata.py`

1. **Place new or updated hardware CSV files in `imudata/`**:
   - Accelerometer CSV columns: `tau (s)` and `<Axis> DUT<k> ADEV (ug)`
   - Gyroscope CSV columns: `tau (s)` and `<Axis> DUT<k> ADEV (deg/h)`
   - If `plot_gyro_avar.m` and `plot_xl_avar.m` are present in `imudata/`, the script automatically parses their `fileNames` and `excludedDuts` lists. Otherwise, it auto-discovers matching `*XL*.csv` and `*Gyro*.csv` files in `imudata/`.

2. **Check the current YAML against `imudata/` (Dry Run)**:
   ```bash
   python3 scripts/calibrate_yaml_from_imudata.py --check-only
   ```
   Prints a per-axis diagnostic table showing the number of DUTs, Mean RMS Error (%), Max Error (%), 1σ Spread Error (%), log-log convexity check (verifying there is no mid-τ hump), and `PASS` / `NEEDS FIT` status. Exits with code `0` if all axes are within tolerance, or `1` if an update is needed.

3. **Check and Automatically Update `config/adis16607.yaml` if Needed**:
   ```bash
   # Updates config/adis16607.yaml in-place if any axis exceeds 1.0% RMS error or has a hump
   python3 scripts/calibrate_yaml_from_imudata.py

   # Custom RMS error tolerance (e.g. 0.5%) and save a .bak backup before modifying
   python3 scripts/calibrate_yaml_from_imudata.py --tolerance-pct 0.5 --backup
   ```

4. **Force a Full Re-Fit and Update of `config/adis16607.yaml`**:
   ```bash
   python3 scripts/calibrate_yaml_from_imudata.py --force-update
   ```
   Fits the 3-pole Gauss-Markov bank (`B_poles`, `correlation_time_poles`), single-pole fallback (`B`, `correlation_time`), `N`, `K`, and part-to-part spread (`N_std`, `B_std`, `K_std`, `correlation_time_std`) across all DUTs and writes the updated parameters into `config/adis16607.yaml`.

---

## Integrating into a Simulation Platform

Once built, you can integrate the IMU error model into a simulation platform in three ways depending on your simulator's architecture:

### Option 1: Sensor Plugin Integration (`imu_nbk_plugin.so`)

Use this when your simulation platform supports the MuJoCo C/C++ sensor plugin interface.

1. **Configure the Scene XML (`my_robot.xml`)**:
   - Declare the `<plugin plugin="mujoco.sensor.imu_nbk"/>` extension.
   - Attach a `<site name="imu_site" .../>` to the robot body where the IMU is mounted.
   - Define the built-in noiseless `<accelerometer>` and `<gyro>` reference sensors **before** the `<plugin>` sensor (the plugin reads their noiseless specific force and angular velocity from `d->sensordata` and adds the full deterministic + multi-pole NBK stochastic error model):
   ```xml
   <mujoco>
     <option timestep="0.001"/>
     <extension>
       <plugin plugin="mujoco.sensor.imu_nbk"/>
     </extension>
     <worldbody>
       <body name="robot_base" pos="0 0 1.0">
         <freejoint/>
         <geom type="box" size="0.1 0.1 0.1" mass="1.0"/>
         <site name="imu_site" pos="0 0 0.1"/>
       </body>
     </worldbody>
     <!-- 1. Noiseless reference sensors — MUST come BEFORE the plugin sensor -->
     <sensor>
       <accelerometer name="acc_true" site="imu_site"/>
       <gyro name="gyro_true" site="imu_site"/>
     </sensor>
     <!-- 2. NBK error model plugin (reads acc_true/gyro_true and outputs 6 noisy channels) -->
     <sensor>
       <plugin name="imu_noisy" plugin="mujoco.sensor.imu_nbk"
               objtype="site" objname="imu_site">
         <config key="config_path" value="config/adis16607.yaml"/>
       </plugin>
     </sensor>
   </mujoco>
   ```

2. **Load the Plugin and Step the Simulation (Python or C/C++)**:
   ```python
   import mujoco

   # Load the compiled plugin shared library once at startup
   mujoco.mj_loadPluginLibrary("build/mujoco_plugin/imu_nbk_plugin.so")

   model = mujoco.MjModel.from_xml_path("my_robot.xml")
   data = mujoco.MjData(model)

   # Step simulation
   mujoco.mj_step(model, data)

   # Sensor layout in data.sensordata:
   #   data.sensordata[0:3]  -> acc_true  (noiseless specific force, m/s^2)
   #   data.sensordata[3:6]  -> gyro_true (noiseless angular rate, rad/s)
   #   data.sensordata[6:9]  -> imu_noisy accelerometer (with NBK errors, m/s^2)
   #   data.sensordata[9:12] -> imu_noisy gyroscope     (with NBK errors, rad/s)
   noisy_acc  = data.sensordata[6:9]
   noisy_gyro = data.sensordata[9:12]

   # Resetting the simulation (automatically triggers the plugin's reset callback)
   mujoco.mj_resetData(model, data)
   ```

---

### Option 2: Direct C++ Library Integration (`libimu_simulator.a`)

Use this when embedding the IMU model directly into a custom C++ simulation platform, physics engine extension, or hardware-in-the-loop (HIL) node.

1. **Link in CMake**:
   ```cmake
   target_link_libraries(my_sim_platform PRIVATE imu_simulator imu_simulator_config)
   ```

2. **Initialize and Step in C++**:
   ```cpp
   #include "config_loader.h"
   #include "imu_simulator.h"

   // 1. Load calibrated ADIS16607-2 parameters from YAML
   imu_simulator::SimulatorConfig cfg =
       imu_simulator::loadConfig("config/adis16607.yaml");

   // 2. If your simulation platform already passes specific force (including gravity)
   //    to the accelerometer, disable the internal WELMEC gravity subtraction:
   cfg.enableSettings.acc.enableLocalRefVec = false;
   cfg.enableSettings.gyro.enableLocalRefVec = false;

   // 3. Construct the simulator (uses nominal 1-DUT parameters + multi-pole GM bank)
   //    Pass cfg.seed (or master_seed + episode_id for Monte Carlo runs)
   imu_simulator::ImuSimulator imu(
       cfg.accSimParams,
       cfg.gyroSimParams,
       cfg.enableSettings,
       cfg.startGeoPositionLlh,
       cfg.startGeoVelocityNed,
       cfg.sampleTime,
       cfg.seed);

   // 4. Inside your simulation platform's 1000 Hz physics loop:
   Eigen::Vector3d a_true_body(0.0, 0.0, 9.797464);     // True specific force in body frame (m/s^2)
   Eigen::Vector3d w_true_body(0.0, 0.0, 0.1);          // True angular velocity in body frame (rad/s)
   Eigen::Quaterniond q_body_to_nav = Eigen::Quaterniond::Identity();

   Eigen::Vector3d noisy_acc =
       imu.generateAccelerationMeasurement(a_true_body, q_body_to_nav);
   Eigen::Vector3d noisy_gyro =
       imu.generateGyroscopeMeasurement(w_true_body, q_body_to_nav);

   // 5. Between episodes / runs:
   imu.reset();
   ```

---

### Option 3: Batched GPU / Python Integration (JAX, NVIDIA Warp, NumPy)

Use this when running batched GPU physics environments or Python-based simulation platforms:

#### JAX Backend (Batched GPU Training — CUDA / ROCm / TPU)

```python
import jax
from imu_nbk import load_config
from imu_nbk.jax_backend import init_state_batched, batched_step, params_to_jax

params = load_config("config/adis16607.yaml")
jax_params = params_to_jax(params)

n_envs = 1024
states = init_state_batched(n_envs)
rng_keys = jax.random.split(jax.random.PRNGKey(0), n_envs)

# In training loop (after stepping physics):
states, noises, rng_keys = batched_step(states, jax_params, rng_keys)
# noises shape: (1024, 6) = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z] per env
```

#### NVIDIA Warp Backend (CUDA Kernel)

```python
from imu_nbk import load_config
from imu_nbk.warp_backend import ImuNbkWarp

params = load_config("config/adis16607.yaml")
sim = ImuNbkWarp(params, n_envs=1024, seed=42)

output = sim.step()                 # wp.array shape (1024, 6) on GPU
numpy_out = sim.get_output_numpy()  # copy to host
```

#### NumPy Backend (CPU Reference / Prototyping)

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
│   └── allan_validation.cpp         # Multi-threaded (OpenMP) static sim → ADEV CSV
├── config/
│   └── adis16607.yaml               # ADIS16607-2 parameters (SI units, multi-pole GM bank)
├── imudata/
│   ├── Lot*_Gyro_*.csv              # Collected gyroscope Allan deviation hardware data
│   ├── Lot*_XL*.csv                 # Collected accelerometer Allan deviation hardware data
│   ├── plot_gyro_avar.m             # MATLAB reference script & excluded-DUT list (gyro)
│   └── plot_xl_avar.m               # MATLAB reference script & excluded-DUT list (accel)
├── scripts/
│   ├── simulate_imu.py              # Unified simulation (plugin/mjx/jax-only)
│   ├── view_simulation.py           # Interactive 3D viewer + live sensor output
│   ├── plot_results.py              # Post-hoc CSV analysis and plotting
│   ├── drift_analysis.py            # Long-duration drift/bias visualization
│   ├── mjx_training_example.py      # RL training loop pattern (1024 envs)
│   ├── allan_variance.py            # Allan deviation simulation plotting
│   ├── calibrate_yaml_from_imudata.py # Checks imudata/ vs. YAML and updates YAML in-place
│   ├── requirements.txt             # Python dependencies (analysis only)
│   └── setup_and_build.sh           # Build helper
├── test/
│   ├── imu_simulator_test.cpp       # C++ unit tests
│   └── golden/                      # Regression reference CSV
└── CMakeLists.txt
```

## Library Architecture

Three-tier design:

- **`imu_simulator`** (C++, Eigen3 only) — core simulation math. Link from MuJoCo plugins or external simulation platforms.
- **`imu_simulator_config`** (C++, adds yaml-cpp) — YAML config loader. Use for standalone tools.
- **`imu_nbk`** (Python) — GPU backends (JAX/Warp) + NumPy reference. Runtime backend selection.

All read the same `config/adis16607.yaml`. No math duplication — every backend implements the identical NBK discrete-time state-space model.

## Model Enhancements vs. Original Baseline (`IMU-Simulator`)

This repository extends the baseline NBK error model from [maximilian-nitsch/IMU-Simulator](https://github.com/maximilian-nitsch/IMU-Simulator) in two key ways to accurately reproduce real-world ADIS16607-2 hardware characterization data over τ ∈ `[0.5, 10⁴] s`:

### 1. Multi-Pole Gauss-Markov Bias Instability Bank (`B_poles`, `correlation_time_poles`)
- **Limitation of the Original Baseline (Single-Pole Gauss-Markov)**:
  - The original simulator approximates `1/f` flicker noise (bias instability `B`) using a **single** first-order Gauss-Markov process per axis:

$$
\dot{z}_B(t) = -\frac{1}{T_c} z_B(t) + w_B(t)
$$

  - In the Allan deviation domain, a single Gauss-Markov pole peaks at a single integration time (`τ ≈ 1.89 · Tc`) and rolls off with slopes `+1/2` (left) and `-1/2` (right).
  - Real MEMS sensors like the ADIS16607-2 exhibit a **wide, flat `1/f` flicker floor** spanning two decades (`τ ∈ [20, 2000] s`). Forcing a single pole to span that wide bowl creates a prominent artificial **"hump" (inflection)** between `τ = 10² s` and `10³ s` before the rate random walk (`K`) takes over.
- **How the Multi-Pole Bank Solves It**:
  - Each axis now supports a bank of up to `kMaxGmPoles = 4` independent first-order Gauss-Markov states (`B_poles` and `correlation_time_poles` in `config/adis16607.yaml`):

$$
b_{\text{GM},k} = \sum_{p=1}^{P} z_{B,p,k}, \qquad z_{B,p,k} = e^{-\Delta t / T_{c,p}} z_{B,p,k-1} + \sigma_{B,p}\sqrt{1 - e^{-2\Delta t / T_{c,p}}}\, w_{p,k}
$$

  - By spacing 3 poles roughly one decade apart in correlation time (`Tc ≈ 15–24 s`, `130–250 s`, and `1300–4300 s`), their Allan variances superpose into a smooth, strictly convex `1/f` floor across the entire mid-τ region.
  - **Effect on the Model**: Eliminates the mid-τ hump completely and reduces the RMS error against the 33-DUT gyroscope and 47-DUT accelerometer hardware curves from `~8–12%` (single-pole) down to **`0.15% – 0.42%` RMS** across `τ ∈ [0.5, 10⁴] s`, with negligible CPU/GPU overhead (2 additional scalar multiply-adds per axis per step).
  - **Backward Compatibility**: If `B_poles` and `correlation_time_poles` are omitted from a YAML config, the simulator automatically falls back to the original single-pole `B` and `correlation_time` parameters.

### 2. Why `ENABLE_DUT_PARAMETER_SPREAD=ON` Exists (Population vs. Single-Unit Simulation)
- **Nominal Mode (`ENABLE_DUT_PARAMETER_SPREAD=OFF`, Baseline Behavior)**:
  - Every seed in a multi-seed run simulates the **exact same nominal sensor** (identical `N`, `B`, `K`, `correlation_time`, and `B_poles`), changing only the random number generator seed (`seed + 0, +1, +2, ...`).
  - At short and mid integration times (`τ < 100 s`), thousands of averaging windows exist within a 10-hour run, so the Allan deviation of a single nominal sensor converges tightly to its theoretical curve — meaning `sim mean + σ` sits almost on top of `sim mean` at short/mid τ and only spreads out at long τ (`τ > 10³ s`) due to finite-run estimation variance.
- **Population Spread Mode (`ENABLE_DUT_PARAMETER_SPREAD=ON`)**:
  - Datasheet and hardware characterization plots (`mean` and `mean + σ` across 33 Gyro DUTs and 47 Accel DUTs) measure a **population of physical sensors** across multiple manufacturing lots, where each unit (DUT) has slight part-to-part variation in white noise (`N_std`), bias instability (`B_std`), random walk (`K_std`), and correlation time (`correlation_time_std`).
  - When compiled with `-DENABLE_DUT_PARAMETER_SPREAD=ON`, each seed in `allan_validation` samples a deterministic virtual DUT from the 1-σ spread parameters in `config/adis16607.yaml` (and scales its multi-pole `B_poles` and `correlation_time_poles` proportionally).
  - **Effect on the Model**: The resulting multi-seed `sim mean + σ` curve captures both **lot-to-lot / part-to-part hardware spread** (visible at short and mid τ) and **stochastic run-to-run variance** (visible at long τ), matching the empirical hardware `mean + σ` upper envelope across the full `τ ∈ [0.5, 10⁴] s` range.

## ADIS16607 Configuration Parameters

Parameters in `config/adis16607.yaml` are stored in SI units and calibrated against real-world ADIS16607-2 Allan variance hardware characterization measurements in `imudata/` (47 accelerometer DUTs and 33 gyroscope DUTs over `τ ∈ [0.5, 10⁴] s`) as well as datasheet specifications:
- **N** (velocity/angular random walk) — determines short-τ slope (`-1/2`)
- **B_poles** and **correlation_time_poles** (Multi-Pole Gauss-Markov Bank, up to 4 poles per axis) — sums multiple independent first-order Gauss-Markov processes spaced roughly one decade apart in correlation time to accurately model a wide, flat, hump-free `1/f` flicker-noise floor across `τ ∈ [10¹, 10³] s`
- **B** and **correlation_time** — single-pole Gauss-Markov fallback parameters (used automatically if `B_poles` / `correlation_time_poles` are omitted from the YAML)
- **K** (acceleration/rate random walk) — determines long-τ slope (`+1/2`)
- **N_std, B_std, K_std, correlation_time_std** — 1-σ part-to-part variation across physical DUTs, used when compiled with `-DENABLE_DUT_PARAMETER_SPREAD=ON`

### Random Seeds & Simulation Platform Integration (Single-DUT Guidance)

#### 1. Single-DUT Behavior in Simulation Platforms vs. Multi-DUT `allan_validation`
When integrating the IMU model (`libimu_simulator.a`) or MuJoCo plugin (`imu_nbk_plugin.so`) into a simulation platform to represent **1 physical/virtual IMU (1 DUT)** on a vehicle or robot:
- **You do not need to recompile with different CMake flags.** Even with `ENABLE_DUT_PARAMETER_SPREAD=ON` (the default build), `ImuSimulator` and `imu_nbk_plugin.so` **never** apply `N_std`, `B_std`, `K_std`, or `correlation_time_std` — they always use the calibrated **nominal (mean)** 1-DUT parameters (`N`, `B_poles`, `correlation_time_poles`, `K`) from `config/adis16607.yaml`.
- Multi-DUT parameter spread (`sampleDutParams`) is only invoked inside `apps/allan_validation.cpp` when running multi-seed population validation (`num_seeds > 1`).
- *(The only scenario where `N_std`, `B_std`, and `K_std` would be sampled in a simulation platform is if you explicitly randomize per-environment parameters during large-scale batched RL domain randomization across thousands of parallel environments.)*

#### 2. Handling Random Seeds (`seed`) in a Simulation Platform
In `ImuSimulator` (and `config/adis16607.yaml`), the `seed` parameter controls **both**:
1. **Turn-on constant errors** (`initializeRandomConstantErrors()`): Draws the fixed turn-on bias (`interval_turn_on_bias`), scale-factor error (`interval_scale_factor`), misalignment (`interval_misalignment`), and non-orthogonality at startup/reset.
2. **In-run stochastic noise** (`generateStochasticErrors()`): Drives the per-step white noise (`N`), multi-pole Gauss-Markov bias instability (`B_poles`), and rate random walk (`K`).

Choose the seed strategy that matches your simulation platform test mode:

| Simulation Platform Use Case | Recommended Seed Setting | Behavior |
| :--- | :--- | :--- |
| **1. Back-to-back algorithm comparison / regression testing** (e.g., comparing two state estimators on the exact same trajectory) | Fixed positive integer (`seed: 42` in YAML, or pass a fixed `seed > 0` to `ImuSimulator`) | **100% deterministic**: Every run (and every `reset()`) produces the exact same turn-on bias and bitwise-identical noise at every timestep. |
| **2. Monte Carlo evaluation across M episodes** (testing robustness across different power-cycles & noise realizations) | Pass `seed = master_seed + episode_id` (e.g., `1000 + episode_id`) when constructing `ImuSimulator` | **Independent but replayable**: Each episode gets a fresh turn-on bias and stochastic noise trajectory, and if episode `#17` fails, you can replay `seed = 1017` to reproduce the exact failure. |
| **3. Live / interactive / unseeded hardware-in-the-loop** (where every power-cycle should be truly random) | `seed: -1` in `config/adis16607.yaml` | Seeds from OS/hardware entropy (`std::random_device`). Every run and every `reset()` draws a new random seed and re-samples turn-on bias and noise. |

> **Multiple IMUs on the same simulated vehicle/robot:** Never initialize two `ImuSimulator` instances on the same platform with the same positive `seed` (e.g., both `seed = 42`), or both sensors will output 100% identical/correlated noise at every timestep. Always offset the seed by sensor index (`seed = base_seed + imu_index`).

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
