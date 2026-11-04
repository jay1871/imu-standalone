#include "config_loader.h"

#include <cmath>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace imu_simulator {

namespace {

Eigen::Vector3d readVector3d(const YAML::Node& node, const std::string& key) {
  auto vec = node[key];
  if (!vec || vec.size() != 3) {
    throw std::runtime_error("Expected 3-element array for key: " + key);
  }
  return Eigen::Vector3d(vec[0].as<double>(), vec[1].as<double>(),
                         vec[2].as<double>());
}

Eigen::Matrix<double, 6, 1> readVector6d(const YAML::Node& node,
                                          const std::string& key) {
  auto vec = node[key];
  if (!vec || vec.size() != 6) {
    throw std::runtime_error("Expected 6-element array for key: " + key);
  }
  Eigen::Matrix<double, 6, 1> result;
  for (int i = 0; i < 6; i++) {
    result(i) = vec[i].as<double>();
  }
  return result;
}

#ifdef ENABLE_DUT_PARAMETER_SPREAD
Eigen::Vector3d readOptionalVector3d(const YAML::Node& node,
                                     const std::string& key) {
  auto vec = node[key];
  if (!vec) {
    return Eigen::Vector3d::Zero();
  }
  if (vec.size() != 3) {
    throw std::runtime_error("Expected 3-element array for key: " + key);
  }
  return Eigen::Vector3d(vec[0].as<double>(), vec[1].as<double>(),
                         vec[2].as<double>());
}
#endif

ImuSimParams readSensorParams(const YAML::Node& node) {
  ImuSimParams params;
  params.N = readVector3d(node, "N");
  params.B = readVector3d(node, "B");
  params.K = readVector3d(node, "K");
  params.corrTime = readVector3d(node, "correlation_time");
#ifdef ENABLE_DUT_PARAMETER_SPREAD
  params.N_std = readOptionalVector3d(node, "N_std");
  params.B_std = readOptionalVector3d(node, "B_std");
  params.K_std = readOptionalVector3d(node, "K_std");
  params.corrTime_std = readOptionalVector3d(node, "correlation_time_std");
#endif
  params.intervalTurnOnBias = readVector3d(node, "interval_turn_on_bias");
  params.measRange = readVector3d(node, "measurement_range");
  params.intervalScaleFactor = readVector3d(node, "interval_scale_factor");
  params.intervalMisAlignment = readVector3d(node, "interval_misalignment");
  params.intervalNonOrthogonality =
      readVector6d(node, "interval_non_orthogonality");
  params.resolution = readVector3d(node, "resolution");
  return params;
}

ModelEnableSettings readEnableSettings(const YAML::Node& node) {
  ModelEnableSettings settings;
  settings.enableLocalRefVec = node["enable_local_ref_vec"].as<bool>();
  settings.enableTurnOnBias = node["enable_turn_on_bias"].as<bool>();
  settings.enableScaling = node["enable_scaling"].as<bool>();
  settings.enableMisAlignment = node["enable_misalignment"].as<bool>();
  settings.enableStochasticError = node["enable_stochastic_error"].as<bool>();
  settings.enableSaturation = node["enable_saturation"].as<bool>();
  settings.enableQuantization = node["enable_quantization"].as<bool>();
  return settings;
}

}  // namespace

SimulatorConfig loadConfig(const std::string& yamlPath) {
  YAML::Node root = YAML::LoadFile(yamlPath);

  auto imu = root["imu_simulator"];
  if (!imu) {
    throw std::runtime_error(
        "YAML missing 'imu_simulator' root key in: " + yamlPath);
  }

  SimulatorConfig config;

  // General settings
  auto general = imu["general_settings"];
  if (!general) {
    throw std::runtime_error(
        "YAML missing 'general_settings' section in: " + yamlPath);
  }
  config.sampleTime = general["sample_time"].as<double>();
  if (config.sampleTime <= 0.0) {
    throw std::runtime_error("sample_time must be > 0 in: " + yamlPath);
  }
  config.seed = general["seed"].as<int>();

  // Model parameters
  auto params = imu["model_parameter_settings"];
  if (!params || !params["accelerometer"] || !params["gyroscope"]) {
    throw std::runtime_error(
        "YAML missing 'model_parameter_settings' (accelerometer/gyroscope) in: " +
        yamlPath);
  }
  config.accSimParams = readSensorParams(params["accelerometer"]);
  config.gyroSimParams = readSensorParams(params["gyroscope"]);

  // Enable settings
  auto enable = imu["model_enable_settings"];
  if (!enable || !enable["accelerometer"] || !enable["gyroscope"]) {
    throw std::runtime_error(
        "YAML missing 'model_enable_settings' (accelerometer/gyroscope) in: " +
        yamlPath);
  }
  config.enableSettings.acc = readEnableSettings(enable["accelerometer"]);
  config.enableSettings.gyro = readEnableSettings(enable["gyroscope"]);

  // Environmental settings
  auto env = imu["environmental_settings"];
  if (!env) {
    throw std::runtime_error(
        "YAML missing 'environmental_settings' section in: " + yamlPath);
  }
  Eigen::Vector3d geoLlhDeg = readVector3d(env, "start_geo_position_llh");
  config.geoPositionLlh = Eigen::Vector3d(
      geoLlhDeg(0) * M_PI / 180.0, geoLlhDeg(1) * M_PI / 180.0, geoLlhDeg(2));
  config.geoVelocityNed = readVector3d(env, "start_geo_velocity_ned");

  return config;
}

ImuSimulator createSimulatorFromConfig(const std::string& yamlPath) {
  SimulatorConfig config = loadConfig(yamlPath);

  ImuSimulator sim(config.accSimParams, config.gyroSimParams,
                   config.enableSettings, config.geoPositionLlh,
                   config.geoVelocityNed, config.sampleTime, config.seed);

  return sim;
}

}  // namespace imu_simulator
