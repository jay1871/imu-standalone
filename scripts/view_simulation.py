#!/usr/bin/env python3
"""Launch MuJoCo interactive viewer with the IMU plugin active.

You see the body moving in 3D while the plugin runs. Press SPACE to
pause, drag to rotate camera, scroll to zoom. Sensor data prints to
terminal in real-time.

Usage:
  python view_simulation.py                              # free-fall body
  python view_simulation.py ../models/spinning_body_imu.xml --ctrl 0.3
"""

import os
import signal
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

signal.signal(signal.SIGINT, lambda *_: os._exit(0))

PLUGIN_PATH = Path(__file__).resolve().parent.parent / "build" / "mujoco_plugin" / "imu_nbk_plugin.so"


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent.parent / "models" / "free_body_imu.xml")
    ctrl_value = None
    if "--ctrl" in sys.argv:
        ctrl_value = float(sys.argv[sys.argv.index("--ctrl") + 1])

    # Load plugin
    if not PLUGIN_PATH.exists():
        sys.exit(f"Plugin not found: {PLUGIN_PATH}\n"
                 f"Build with: cmake .. -DIMU_BUILD_MUJOCO_PLUGIN=ON ...")
    mujoco.mj_loadPluginLibrary(str(PLUGIN_PATH))

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    # Sensor addresses
    acc_true_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "acc_true")
    gyro_true_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro_true")
    imu_noisy_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_noisy")

    acc_true_adr = m.sensor_adr[acc_true_id]
    gyro_true_adr = m.sensor_adr[gyro_true_id]
    imu_noisy_adr = m.sensor_adr[imu_noisy_id]

    speed = 1.0
    paused = False

    print(f"Model: {model_path}")
    print(f"Timestep: {m.opt.timestep*1000:.1f} ms")
    print(f"Plugin loaded: {PLUGIN_PATH.name}")
    print()
    print("Controls:")
    print("  SPACE     = pause/resume")
    print("  UP / ]    = speed up 2x")
    print("  DOWN / [  = slow down 2x")
    print("  R         = reset simulation")
    print("  Q / ESC   = quit")
    print("-" * 70)
    print(f"{'Time':>6s}  {'Speed':>5s}  {'Acc True (m/s²)':^30s}  {'Acc Noisy (m/s²)':^30s}")
    print("-" * 70)

    step_count = 0
    wall_start = time.perf_counter()
    sim_start = 0.0

    def key_callback(key):
        nonlocal speed, paused, wall_start, sim_start
        # key is a glfw key code
        # SPACE=32, UP=265, DOWN=264, R=82, Q=81, ESC=256, ]=93, [=91
        if key == 32:  # SPACE
            paused = not paused
            if not paused:
                wall_start = time.perf_counter()
                sim_start = d.time
            print(f"  {'PAUSED' if paused else f'RUNNING (speed={speed:.2f}x)'}")
        elif key in (265, 93):  # UP or ]
            speed = min(speed * 2.0, 64.0)
            wall_start = time.perf_counter()
            sim_start = d.time
            print(f"  Speed: {speed:.2f}x")
        elif key in (264, 91):  # DOWN or [
            speed = max(speed / 2.0, 0.0625)
            wall_start = time.perf_counter()
            sim_start = d.time
            print(f"  Speed: {speed:.2f}x")
        elif key == 82:  # R
            mujoco.mj_resetData(m, d)
            wall_start = time.perf_counter()
            sim_start = 0.0
            print("  RESET")
        elif key in (81, 256):  # Q or ESC
            sys.exit(0)

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if paused:
                viewer.sync()
                time.sleep(0.02)
                continue

            if ctrl_value is not None and m.nu > 0:
                d.ctrl[0] = ctrl_value

            mujoco.mj_step(m, d)
            step_count += 1

            # Print every 100 steps (~0.1s at 1kHz)
            if step_count % 100 == 0:
                at = d.sensordata[acc_true_adr:acc_true_adr + 3]
                an = d.sensordata[imu_noisy_adr:imu_noisy_adr + 3]
                print(f"{d.time:6.2f}  {speed:5.2f}  "
                      f"[{at[0]:+8.4f} {at[1]:+8.4f} {at[2]:+8.4f}]  "
                      f"[{an[0]:+8.4f} {an[1]:+8.4f} {an[2]:+8.4f}]")

            # Sync viewer at ~60 Hz (not every physics step)
            if step_count % max(1, int(1.0 / (60.0 * m.opt.timestep))) == 0:
                viewer.sync()

            # Real-time pacing with speed factor
            sim_elapsed = d.time - sim_start
            wall_elapsed = time.perf_counter() - wall_start
            sleep_time = (sim_elapsed / speed) - wall_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
