/*@license BSD-3 https://opensource.org/licenses/BSD-3-Clause
Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
Maximilian Nitsch (m.nitsch@irt.rwth-aachen.de)
All rights reserved.
*/

#pragma once

#include <Eigen/Dense>

// Uncomment the #define below (or configure CMake with -DENABLE_DUT_PARAMETER_SPREAD=ON)
// to enable multi-DUT part-to-part parameter spread (N_std, B_std, K_std, correlation_time_std)
// derived from real-world hardware characterization measurements.
// #define ENABLE_DUT_PARAMETER_SPREAD

namespace imu_simulator {

constexpr int kMaxGmPoles = 4;

struct ImuStochasticErrors {
  Eigen::Vector3d z_N = Eigen::Vector3d::Zero();  // Velocity/angular random walk
  Eigen::Vector3d z_B = Eigen::Vector3d::Zero();  // Total bias instability (sum of poles)
  Eigen::Vector3d z_K = Eigen::Vector3d::Zero();  // Accel./rate random walk
  Eigen::Matrix<double, 3, kMaxGmPoles> z_B_poles =
      Eigen::Matrix<double, 3, kMaxGmPoles>::Zero();  // Per-pole Gauss-Markov bias states
};

struct ImuSimParams {
  Eigen::Vector3d N = Eigen::Vector3d::Zero();  // Vel./ang. random walk (m/s^(3/2))/(rad/s^(1/2))
  Eigen::Vector3d B = Eigen::Vector3d::Zero();  // Bias instability (m/s^2)/(rad/s)
  Eigen::Vector3d K = Eigen::Vector3d::Zero();  // Accel./rate random walk (m/s^(5/2)))/(rad/s^(3/2)))
  Eigen::Vector3d corrTime = Eigen::Vector3d::Constant(100.0);  // Correlation time (s)
  int numGmPoles = 1;  // Number of Gauss-Markov poles per axis (1 = standard single-pole, up to kMaxGmPoles)
  Eigen::Matrix<double, 3, kMaxGmPoles> B_poles =
      Eigen::Matrix<double, 3, kMaxGmPoles>::Zero();  // Multi-pole bias instability (rows: X,Y,Z; cols: poles)
  Eigen::Matrix<double, 3, kMaxGmPoles> corrTime_poles =
      Eigen::Matrix<double, 3, kMaxGmPoles>::Constant(100.0);  // Multi-pole correlation times (s)
#ifdef ENABLE_DUT_PARAMETER_SPREAD
  Eigen::Vector3d N_std = Eigen::Vector3d::Zero();         // Optional DUT-to-DUT 1-sigma spread for N
  Eigen::Vector3d B_std = Eigen::Vector3d::Zero();         // Optional DUT-to-DUT 1-sigma spread for B
  Eigen::Vector3d K_std = Eigen::Vector3d::Zero();         // Optional DUT-to-DUT 1-sigma spread for K
  Eigen::Vector3d corrTime_std = Eigen::Vector3d::Zero();  // Optional DUT-to-DUT 1-sigma spread for corrTime
#endif
  Eigen::Vector3d intervalTurnOnBias = Eigen::Vector3d::Zero();  // Interval turn-on bias (m/s^2)/(rad/s)
  Eigen::Vector3d measRange = Eigen::Vector3d::Zero();           // Measurement range (m/s^2)/(rad/s)
  Eigen::Vector3d intervalMisAlignment = Eigen::Vector3d::Zero();  // Interval misalignment (mrad)
  Eigen::Matrix<double, 6, 1> intervalNonOrthogonality =
      Eigen::Matrix<double, 6, 1>::Zero();  // Interval nonOrthogonality (mrad) //NOLINT
  Eigen::Vector3d intervalScaleFactor = Eigen::Vector3d::Zero();  // Interval scale factor (%)
  Eigen::Vector3d resolution = Eigen::Vector3d::Zero();  // Measurement resolution (m/s^2/LSB)/(rad/s/LSB)
};

struct ModelEnableSettings {
  bool enableLocalRefVec = false;      // on/off gravity (acc)/ Earth angular vel. (gyro)
  bool enableTurnOnBias = false;       // on/off constant turn-on bias given an interval
  bool enableScaling = false;          // on/off scaling
  bool enableMisAlignment = false;     // on/off misalignment
  bool enableStochasticError = false;  // on/off colored noise (stochastic error)
  bool enableSaturation = false;       // on/off saturation given a measurement range
  bool enableQuantization = false;     // on/off quantization given a resolution
};

struct ImuModelEnableSettings {
  ModelEnableSettings acc;
  ModelEnableSettings gyro;
};

struct EnvironmentalParameters {
  Eigen::Vector3d g_n = Eigen::Vector3d::Zero();     // Local gravity vector (m/s²)
  Eigen::Vector3d w_ie_n = Eigen::Vector3d::Zero();  // Local angular velocity of Earth (rad/s)
};

// Identifier for measurement axis XYZ
enum AxisIdentifier : int { axisX = 0, axisY = 1, axisZ = 2 };

// Identifier for measurement type (accelerometer or gyroscope)
enum MeasurementType { acc, gyro };

}  // namespace imu_simulator
