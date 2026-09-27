"""MuJoCo plugin integration tests.

These tests require the MuJoCo C sensor plugin to be built and the mujoco
Python package installed. They verify the plugin produces correct sensor
output when loaded into a MuJoCo simulation.

Skip automatically if mujoco is not available.
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture(autouse=True)
def _skip_if_no_mujoco():
    pytest.importorskip("mujoco")


@pytest.fixture
def plugin_path():
    """Path to the compiled plugin shared library."""
    build_dir = Path(__file__).parent.parent.parent / "build" / "mujoco_plugin"
    for ext in [".so", ".dll", ".dylib"]:
        path = build_dir / f"imu_nbk_plugin{ext}"
        if path.exists():
            return str(path)
    pytest.skip("MuJoCo plugin not built (run cmake with -DIMU_BUILD_MUJOCO_PLUGIN=ON)")


@pytest.fixture
def config_path():
    """Path to the shared YAML config."""
    return str(Path(__file__).parent.parent.parent / "config" / "adis16607.yaml")


@pytest.fixture
def model_xml(config_path):
    """Minimal MuJoCo XML model with reference sensors and the IMU plugin sensor."""
    return f"""
    <mujoco>
      <extension>
        <plugin plugin="mujoco.sensor.imu_nbk"/>
      </extension>
      <worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <body name="imu_body" pos="0 0 0.05">
          <freejoint/>
          <geom type="box" size="0.05 0.05 0.05" mass="0.1"/>
          <site name="imu_site" pos="0 0 0"/>
        </body>
      </worldbody>
      <sensor>
        <accelerometer name="acc_true" site="imu_site"/>
        <gyro name="gyro_true" site="imu_site"/>
        <plugin name="imu_nbk" plugin="mujoco.sensor.imu_nbk"
                objtype="site" objname="imu_site">
          <config key="config_path" value="{config_path}"/>
        </plugin>
      </sensor>
    </mujoco>
    """


def get_plugin_sensordata(model, data):
    import mujoco
    plugin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_nbk")
    adr = model.sensor_adr[plugin_id]
    return data.sensordata[adr:adr + 6]


class TestPluginLoads:
    """Basic plugin loading and initialization."""

    def test_plugin_loads(self, plugin_path, model_xml):
        """Plugin library loads and model compiles without error."""
        import mujoco

        mujoco.mj_loadPluginLibrary(plugin_path)

        model = mujoco.MjModel.from_xml_string(model_xml)
        data = mujoco.MjData(model)
        assert model.nsensor == 3
        plugin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_nbk")
        assert model.sensor_dim[plugin_id] == 6

    def test_plugin_produces_output(self, plugin_path, model_xml):
        """Plugin writes non-zero sensor data after stepping."""
        import mujoco

        mujoco.mj_loadPluginLibrary(plugin_path)
        model = mujoco.MjModel.from_xml_string(model_xml)
        data = mujoco.MjData(model)

        # Step a few times
        for _ in range(10):
            mujoco.mj_step(model, data)

        sensor_data = get_plugin_sensordata(model, data).copy()
        # At least some axes should have non-zero output (stochastic noise)
        assert np.any(sensor_data != 0.0), "Plugin produced all-zero output"

    def test_plugin_reset(self, plugin_path, model_xml):
        """Plugin state resets properly."""
        import mujoco

        mujoco.mj_loadPluginLibrary(plugin_path)
        model = mujoco.MjModel.from_xml_string(model_xml)
        data = mujoco.MjData(model)

        # Step several times to build up state
        for _ in range(100):
            mujoco.mj_step(model, data)

        # Reset
        mujoco.mj_resetData(model, data)
        mujoco.mj_step(model, data)

        sensor_after_reset = get_plugin_sensordata(model, data).copy()
        assert np.all(np.isfinite(sensor_after_reset))


class TestPluginPhysics:
    """Plugin correctly reads physics state."""

    def test_gravity_affects_accel(self, plugin_path, model_xml):
        """With gravity enabled, accelerometer should measure ~g on z-axis."""
        import mujoco

        mujoco.mj_loadPluginLibrary(plugin_path)
        model = mujoco.MjModel.from_xml_string(model_xml)
        data = mujoco.MjData(model)

        # Step simulation to let body settle on ground plane
        for _ in range(50):
            mujoco.mj_step(model, data)

        plugin_data = get_plugin_sensordata(model, data)
        acc_z = plugin_data[2]
        # Accelerometer should measure reaction force to gravity (~9.8 m/s^2)
        assert abs(acc_z) > 5.0, f"Accel Z = {acc_z}, expected ~9.8"
