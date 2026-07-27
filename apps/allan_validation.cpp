/*
 * Allan Variance Validation Tool
 *
 * Runs a static IMU simulation and outputs time-series data to CSV
 * for Allan variance analysis. Only stochastic errors + gravity/earth-rate
 * are enabled so the Allan deviation cleanly reflects N, B, K parameters.
 *
 * Usage:
 *   ./allan_validation <config.yaml> [output.csv] [duration_hours] [sample_rate_hz]
 *
 * Default: 10 hours at 1000 Hz using config as specified
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "config_loader.h"

int main(int argc, char** argv) {
  std::string configPath = "config/adis16607.yaml";
  std::string outputPath = "allan_validation_data.csv";
  double durationHours = 10.0;
  double sampleRateHz = 0.0;  // 0 = use config sample_time

  if (argc > 1) configPath = argv[1];
  if (argc > 2) outputPath = argv[2];
  if (argc > 3) durationHours = std::atof(argv[3]);
  if (argc > 4) sampleRateHz = std::atof(argv[4]);

  std::cout << "Loading config: " << configPath << std::endl;

  imu_simulator::SimulatorConfig config =
      imu_simulator::loadConfig(configPath);

  // Override sample rate if specified on command line
  if (sampleRateHz > 0.0) {
    config.sampleTime = 1.0 / sampleRateHz;
  }

  double dt = config.sampleTime;
  double rate = 1.0 / dt;

  // For Allan validation: disable deterministic errors, keep only stochastic + reference
  config.enableSettings.acc.enableTurnOnBias = false;
  config.enableSettings.acc.enableScaling = false;
  config.enableSettings.acc.enableMisAlignment = false;
  config.enableSettings.acc.enableStochasticError = true;
  config.enableSettings.acc.enableSaturation = false;
  config.enableSettings.acc.enableQuantization = false;
  config.enableSettings.acc.enableLocalRefVec = true;

  config.enableSettings.gyro.enableTurnOnBias = false;
  config.enableSettings.gyro.enableScaling = false;
  config.enableSettings.gyro.enableMisAlignment = false;
  config.enableSettings.gyro.enableStochasticError = true;
  config.enableSettings.gyro.enableSaturation = false;
  config.enableSettings.gyro.enableQuantization = false;
  config.enableSettings.gyro.enableLocalRefVec = true;

  imu_simulator::ImuSimulator sim(
      config.accSimParams, config.gyroSimParams, config.enableSettings,
      config.geoPositionLlh, config.geoVelocityNed, dt, config.seed);

  sim.setUseFixedRandomNumbersFlag(false);
  sim.resetImuSimulator();

  // Static simulation: zero true motion, identity orientation
  Eigen::Vector3d a_true = Eigen::Vector3d::Zero();
  Eigen::Vector3d w_true = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_identity(1.0, 0.0, 0.0, 0.0);

  double totalSeconds = durationHours * 3600.0;
  long totalSamples = static_cast<long>(totalSeconds / dt);

  std::cout << "Allan Variance Validation" << std::endl;
  std::cout << "  Config:      " << configPath << std::endl;
  std::cout << "  Output:      " << outputPath << std::endl;
  std::cout << "  Duration:    " << durationHours << " hours" << std::endl;
  std::cout << "  Sample rate: " << rate << " Hz" << std::endl;
  std::cout << "  Total samples: " << totalSamples << std::endl;
  std::cout << "  Estimated file size: "
            << (totalSamples * 80) / (1024 * 1024) << " MB" << std::endl;
  std::cout << std::endl;

  std::ofstream outFile(outputPath);
  if (!outFile.is_open()) {
    std::cerr << "Error: cannot open output file: " << outputPath << std::endl;
    return 1;
  }

  outFile << "time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z" << std::endl;
  outFile << std::setprecision(12);

  auto startTime = std::chrono::steady_clock::now();
  int progressPercent = 0;

  for (long i = 0; i < totalSamples; i++) {
    Eigen::Vector3d acc = sim.generateAccelerationMeasurement(a_true, q_identity);
    Eigen::Vector3d gyro = sim.generateGyroscopeMeasurement(w_true, q_identity);

    double t = i * dt;
    outFile << t << ","
            << acc(0) << "," << acc(1) << "," << acc(2) << ","
            << gyro(0) << "," << gyro(1) << "," << gyro(2) << "\n";

    // Progress reporting every 1%
    int pct = static_cast<int>((100.0 * i) / totalSamples);
    if (pct > progressPercent) {
      progressPercent = pct;
      auto now = std::chrono::steady_clock::now();
      double elapsed =
          std::chrono::duration<double>(now - startTime).count();
      double eta = (elapsed / pct) * (100 - pct);
      std::cout << "\r  Progress: " << pct << "% | Elapsed: "
                << static_cast<int>(elapsed) << "s | ETA: "
                << static_cast<int>(eta) << "s" << std::flush;
    }
  }

  outFile.close();

  auto endTime = std::chrono::steady_clock::now();
  double totalElapsed =
      std::chrono::duration<double>(endTime - startTime).count();

  std::cout << "\r  Progress: 100% | Total time: "
            << static_cast<int>(totalElapsed) << "s" << std::endl;
  std::cout << std::endl;
  std::cout << "Done! Output written to: " << outputPath << std::endl;
  std::cout << std::endl;
  std::cout << "Next step: run Allan variance analysis:" << std::endl;
  std::cout << "  python3 scripts/allan_variance.py " << outputPath << " "
            << configPath << " --rate " << rate << std::endl;

  return 0;
}
