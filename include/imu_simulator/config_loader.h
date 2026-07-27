/*
 * Config loader for IMU Simulator - parses YAML into simulator parameter structs.
 * Depends on yaml-cpp. The core imu_simulator library does NOT depend on this.
 */

#pragma once

#include <string>

#include "imu_simulator.h"
#include "imu_simulator_structures.h"

namespace imu_simulator {

struct SimulatorConfig {
  ImuSimParams accSimParams;
  ImuSimParams gyroSimParams;
  ImuModelEnableSettings enableSettings;
  double sampleTime;
  unsigned int seed;
  bool useConstantSeed;
  Eigen::Vector3d geoPositionLlh;
  Eigen::Vector3d geoVelocityNed;
};

SimulatorConfig loadConfig(const std::string& yamlPath);

ImuSimulator createSimulatorFromConfig(const std::string& yamlPath);

}  // namespace imu_simulator
