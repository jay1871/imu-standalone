# IMU Simulator Standalone

Standalone C++ IMU error model simulator extracted from [maximilian-nitsch/IMU-Simulator](https://github.com/maximilian-nitsch/IMU-Simulator). No ROS2 dependencies. Configured for the ADIS16607-2 with Allan variance validation.

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

### Python Analysis (for Allan variance plotting)

| Package | Purpose |
|---------|---------|
| numpy >= 1.20 | Numerical arrays |
| matplotlib >= 3.5 | Plotting |
| allantools >= 2019.9 | Allan deviation computation |
| pyyaml >= 5.0 | YAML config reading |

Install:
```bash
pip install -r scripts/requirements.txt
```

## Build

```bash
cd imu-standalone
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
ctest --output-on-failure
```

Or use the helper script:
```bash
bash scripts/setup_and_build.sh
```

## Usage

### Allan Variance Validation

Run from the `build/` directory:

```bash
cd build

# Quick run (1 hour, fast iteration)
./allan_validation ../config/adis16607.yaml allan_data.csv 1 1000

# Full run (10 hours, proper long-tau resolution)
./allan_validation ../config/adis16607.yaml allan_data.csv 10 1000

# Generate plots
python3 ../scripts/allan_variance.py allan_data.csv ../config/adis16607.yaml --rate 1000
```

### Command-line arguments

```
./build/allan_validation <config.yaml> [output.csv] [duration_hours] [sample_rate_hz]
```

| Argument | Default | Description |
|----------|---------|-------------|
| config.yaml | (required) | Path to YAML parameter file |
| output.csv | `allan_validation_data.csv` | Output CSV path |
| duration_hours | 10 | Simulation duration |
| sample_rate_hz | from config (1000) | Override sample rate |

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
├── apps/
│   └── allan_validation.cpp         # Static sim → CSV
├── config/
│   └── adis16607.yaml               # ADIS16607-2 parameters (SI units)
├── scripts/
│   ├── allan_variance.py            # Allan deviation analysis + plotting
│   ├── requirements.txt             # Python dependencies
│   └── setup_and_build.sh           # Build helper
└── test/                            # GTest unit tests
```

## Library Architecture

Two libraries for flexible linking:

- **`imu_simulator`** — core simulation, depends only on Eigen3. Link this from MuJoCo/Isaac Sim plugins.
- **`imu_simulator_config`** — YAML config loader, adds yaml-cpp. Use for standalone tools.

## ADIS16607 Configuration

Parameters in `config/adis16607.yaml` are converted to SI units from the ADIS16607 datasheet Table 1. Parameters not available from the datasheet are marked with `ESTIMATE` comments — update these after running Allan variance characterization on real hardware:

```bash
grep "ESTIMATE" config/adis16607.yaml
```

These are:
- **K** (rate/acceleration random walk) — determines long-tau slope (+1/2)
- **correlation_time** — Gauss-Markov time constant, affects bias instability plateau location

## References

- Nitsch, M. et al. — [IMU-Simulator](https://github.com/maximilian-nitsch/IMU-Simulator)
- Farrell, J. et al. — "Inertial Measurement Unit Error Modeling Tutorial", IEEE Control Systems Magazine, 2022
- ADIS16607 Datasheet, Analog Devices
