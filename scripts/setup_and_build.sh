#!/bin/bash
# Setup and build script for imu-standalone
# Run from the imu-standalone/ directory

set -e

echo "=== IMU Simulator Standalone Build ==="
echo ""

# Check dependencies
echo "Checking dependencies..."
MISSING=""

if ! dpkg -s libeigen3-dev > /dev/null 2>&1; then
    MISSING="$MISSING libeigen3-dev"
fi
if ! dpkg -s libyaml-cpp-dev > /dev/null 2>&1; then
    MISSING="$MISSING libyaml-cpp-dev"
fi
if ! dpkg -s libgtest-dev > /dev/null 2>&1; then
    MISSING="$MISSING libgtest-dev"
fi
if ! dpkg -s cmake > /dev/null 2>&1; then
    MISSING="$MISSING cmake"
fi

if [ -n "$MISSING" ]; then
    echo "Missing packages:$MISSING"
    echo ""
    echo "Install with:"
    echo "  sudo apt-get install -y$MISSING"
    echo ""
    read -p "Install now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt-get install -y $MISSING
    else
        echo "Please install dependencies and re-run."
        exit 1
    fi
fi

echo "All dependencies found."
echo ""

# Build
echo "Configuring with CMake..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_APPS=ON

echo ""
echo "Building..."
cmake --build . -j$(nproc)

echo ""
echo "Running tests..."
ctest --output-on-failure

echo ""
echo "=== Build successful! ==="
echo ""
echo "Executables:"
echo "  ./build/allan_validation"
echo ""
echo "Quick validation run (1 hour):"
echo "  ./build/allan_validation config/adis16607.yaml allan_data.csv 1 1000"
echo ""
echo "Full validation run (10 hours):"
echo "  ./build/allan_validation config/adis16607.yaml allan_data.csv 10 1000"
echo ""
echo "Python analysis:"
echo "  pip install -r scripts/requirements.txt"
echo "  python3 scripts/allan_variance.py allan_data.csv config/adis16607.yaml --rate 1000"
