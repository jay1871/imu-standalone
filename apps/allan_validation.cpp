/*
 * Allan Variance Validation Tool
 *
 * Runs static IMU simulation(s) and computes overlapping Allan deviation
 * in-memory — no multi-GB CSV files needed.
 *
 * Single seed: outputs one ADEV curve per axis (tau, adev values).
 * Multi seed:  runs N independent realizations and outputs mean ± σ envelope.
 *
 * Usage:
 *   ./allan_validation <config.yaml> [output.csv] [duration_hours] [sample_rate_hz] [num_seeds]
 *
 * Examples:
 *   ./allan_validation ../config/adis16607.yaml adev.csv 10 1000        # single seed
 *   ./allan_validation ../config/adis16607.yaml adev.csv 10 1000 100    # 100 seeds (mean±σ)
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "config_loader.h"

struct AdevResult {
  std::vector<double> taus;
  std::vector<double> acc_adev[3];
  std::vector<double> gyro_adev[3];
};

std::vector<double> computeOctaveTaus(long totalSamples, double dt) {
  std::vector<double> taus;
  double tau_min = dt;
  double tau_max = totalSamples * dt / 3.0;
  double tau = tau_min;
  while (tau <= tau_max) {
    taus.push_back(tau);
    tau *= 2.0;
  }
  return taus;
}

std::vector<double> computeOverlappingAdev(const std::vector<double>& data,
                                            double dt,
                                            const std::vector<double>& taus) {
  long N = static_cast<long>(data.size());
  std::vector<double> adev_out(taus.size(), 0.0);

  for (size_t ti = 0; ti < taus.size(); ti++) {
    long m = static_cast<long>(std::round(taus[ti] / dt));
    if (m < 1) m = 1;
    if (2 * m >= N) {
      adev_out[ti] = 0.0;
      continue;
    }

    double sum = 0.0;
    long count = 0;
    for (long i = 0; i + 2 * m <= N; i++) {
      double diff = data[i + 2 * m] - 2.0 * data[i + m] + data[i];
      sum += diff * diff;
      count++;
    }

    if (count > 0) {
      double tau_val = m * dt;
      adev_out[ti] = std::sqrt(sum / (2.0 * tau_val * tau_val * count));
    }
  }
  return adev_out;
}

AdevResult runSingleSeed(imu_simulator::SimulatorConfig& config, double dt,
                          long totalSamples, int seed,
                          const std::vector<double>& taus) {
  imu_simulator::ImuSimulator sim(config.accSimParams, config.gyroSimParams,
                                   config.enableSettings, config.geoPositionLlh,
                                   config.geoVelocityNed, dt, seed);
  sim.resetImuSimulator();

  Eigen::Vector3d a_true = Eigen::Vector3d::Zero();
  Eigen::Vector3d w_true = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_identity(1.0, 0.0, 0.0, 0.0);

  // Phase-integrated output (Allan deviation of frequency data = use raw values)
  std::vector<double> acc_data[3];
  std::vector<double> gyro_data[3];
  for (int ax = 0; ax < 3; ax++) {
    acc_data[ax].reserve(totalSamples);
    gyro_data[ax].reserve(totalSamples);
  }

  // Generate time series in memory
  for (long i = 0; i < totalSamples; i++) {
    Eigen::Vector3d acc = sim.generateAccelerationMeasurement(a_true, q_identity);
    Eigen::Vector3d gyro = sim.generateGyroscopeMeasurement(w_true, q_identity);

    // Store phase (integrated) for OADEV — cumulative sum × dt
    for (int ax = 0; ax < 3; ax++) {
      double prev_acc = acc_data[ax].empty() ? 0.0 : acc_data[ax].back();
      double prev_gyro = gyro_data[ax].empty() ? 0.0 : gyro_data[ax].back();
      acc_data[ax].push_back(prev_acc + acc(ax) * dt);
      gyro_data[ax].push_back(prev_gyro + gyro(ax) * dt);
    }
  }

  // Compute OADEV
  AdevResult result;
  result.taus = taus;
  for (int ax = 0; ax < 3; ax++) {
    result.acc_adev[ax] = computeOverlappingAdev(acc_data[ax], dt, taus);
    result.gyro_adev[ax] = computeOverlappingAdev(gyro_data[ax], dt, taus);
  }

  return result;
}

int main(int argc, char** argv) {
  std::string configPath = "config/adis16607.yaml";
  std::string outputPath = "allan_adev.csv";
  double durationHours = 10.0;
  double sampleRateHz = 0.0;
  int numSeeds = 1;

  if (argc > 1) configPath = argv[1];
  if (argc > 2) outputPath = argv[2];
  if (argc > 3) durationHours = std::atof(argv[3]);
  if (argc > 4) sampleRateHz = std::atof(argv[4]);
  if (argc > 5) numSeeds = std::atoi(argv[5]);

  std::cout << "Loading config: " << configPath << std::endl;

  imu_simulator::SimulatorConfig config =
      imu_simulator::loadConfig(configPath);

  if (sampleRateHz > 0.0) {
    config.sampleTime = 1.0 / sampleRateHz;
  }

  double dt = config.sampleTime;
  double rate = 1.0 / dt;

  // Only stochastic errors enabled — disable everything else.
  // Gravity/earth-rate models add large DC offsets that cause floating-point
  // precision loss when integrated into phase data for ADEV computation.
  config.enableSettings.acc.enableTurnOnBias = false;
  config.enableSettings.acc.enableScaling = false;
  config.enableSettings.acc.enableMisAlignment = false;
  config.enableSettings.acc.enableStochasticError = true;
  config.enableSettings.acc.enableSaturation = false;
  config.enableSettings.acc.enableQuantization = false;
  config.enableSettings.acc.enableLocalRefVec = false;

  config.enableSettings.gyro.enableTurnOnBias = false;
  config.enableSettings.gyro.enableScaling = false;
  config.enableSettings.gyro.enableMisAlignment = false;
  config.enableSettings.gyro.enableStochasticError = true;
  config.enableSettings.gyro.enableSaturation = false;
  config.enableSettings.gyro.enableQuantization = false;
  config.enableSettings.gyro.enableLocalRefVec = false;

  double totalSeconds = durationHours * 3600.0;
  long totalSamples = static_cast<long>(totalSeconds / dt);
  double memPerRun = (totalSamples * 6.0 * 8.0) / (1024.0 * 1024.0);

  std::cout << "Allan Deviation Validation (in-memory)" << std::endl;
  std::cout << "  Config:       " << configPath << std::endl;
  std::cout << "  Output:       " << outputPath << std::endl;
  std::cout << "  Duration:     " << durationHours << " hours" << std::endl;
  std::cout << "  Sample rate:  " << rate << " Hz" << std::endl;
  std::cout << "  Total samples:" << totalSamples << std::endl;
  std::cout << "  RAM per run:  " << static_cast<int>(memPerRun) << " MB" << std::endl;
  std::cout << "  Seeds:        " << numSeeds << std::endl;
  std::cout << std::endl;

  auto taus = computeOctaveTaus(totalSamples, dt);
  size_t numTaus = taus.size();

  // Accumulators for mean and variance (Welford's online algorithm)
  std::vector<double> acc_mean[3], acc_m2[3];
  std::vector<double> gyro_mean[3], gyro_m2[3];
  for (int ax = 0; ax < 3; ax++) {
    acc_mean[ax].assign(numTaus, 0.0);
    acc_m2[ax].assign(numTaus, 0.0);
    gyro_mean[ax].assign(numTaus, 0.0);
    gyro_m2[ax].assign(numTaus, 0.0);
  }

  auto totalStart = std::chrono::steady_clock::now();

  for (int s = 0; s < numSeeds; s++) {
    int seed = config.seed + s;

    auto runStart = std::chrono::steady_clock::now();

    std::cout << "\r  Seed " << (s + 1) << "/" << numSeeds
              << " (seed=" << seed << ") — simulating..." << std::flush;

    AdevResult result = runSingleSeed(config, dt, totalSamples, seed, taus);

    // Welford update
    int n = s + 1;
    for (int ax = 0; ax < 3; ax++) {
      for (size_t ti = 0; ti < numTaus; ti++) {
        double val_a = result.acc_adev[ax][ti];
        double delta_a = val_a - acc_mean[ax][ti];
        acc_mean[ax][ti] += delta_a / n;
        acc_m2[ax][ti] += delta_a * (val_a - acc_mean[ax][ti]);

        double val_g = result.gyro_adev[ax][ti];
        double delta_g = val_g - gyro_mean[ax][ti];
        gyro_mean[ax][ti] += delta_g / n;
        gyro_m2[ax][ti] += delta_g * (val_g - gyro_mean[ax][ti]);
      }
    }

    auto runEnd = std::chrono::steady_clock::now();
    double runElapsed = std::chrono::duration<double>(runEnd - runStart).count();
    double totalElapsed = std::chrono::duration<double>(runEnd - totalStart).count();
    double eta = (totalElapsed / n) * (numSeeds - n);

    std::cout << "\r  Seed " << (s + 1) << "/" << numSeeds
              << " done (" << static_cast<int>(runElapsed) << "s)"
              << " | Total: " << static_cast<int>(totalElapsed) << "s"
              << " | ETA: " << static_cast<int>(eta) << "s     " << std::flush;
  }
  std::cout << std::endl;

  // Write output CSV
  std::ofstream outFile(outputPath);
  if (!outFile.is_open()) {
    std::cerr << "Error: cannot open output file: " << outputPath << std::endl;
    return 1;
  }

  outFile << std::setprecision(12);

  if (numSeeds == 1) {
    outFile << "tau,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z" << std::endl;
    for (size_t ti = 0; ti < numTaus; ti++) {
      outFile << taus[ti];
      for (int ax = 0; ax < 3; ax++) outFile << "," << acc_mean[ax][ti];
      for (int ax = 0; ax < 3; ax++) outFile << "," << gyro_mean[ax][ti];
      outFile << "\n";
    }
  } else {
    outFile << "tau"
            << ",acc_x_mean,acc_x_std,acc_y_mean,acc_y_std,acc_z_mean,acc_z_std"
            << ",gyro_x_mean,gyro_x_std,gyro_y_mean,gyro_y_std,gyro_z_mean,gyro_z_std"
            << std::endl;
    for (size_t ti = 0; ti < numTaus; ti++) {
      outFile << taus[ti];
      for (int ax = 0; ax < 3; ax++) {
        double std_val = (numSeeds > 1) ? std::sqrt(acc_m2[ax][ti] / (numSeeds - 1)) : 0.0;
        outFile << "," << acc_mean[ax][ti] << "," << std_val;
      }
      for (int ax = 0; ax < 3; ax++) {
        double std_val = (numSeeds > 1) ? std::sqrt(gyro_m2[ax][ti] / (numSeeds - 1)) : 0.0;
        outFile << "," << gyro_mean[ax][ti] << "," << std_val;
      }
      outFile << "\n";
    }
  }

  outFile.close();

  auto totalEnd = std::chrono::steady_clock::now();
  double totalTime = std::chrono::duration<double>(totalEnd - totalStart).count();

  std::cout << std::endl;
  std::cout << "Done! Total time: " << static_cast<int>(totalTime) << "s" << std::endl;
  std::cout << "Output: " << outputPath << " (" << numTaus << " tau points)" << std::endl;
  std::cout << std::endl;
  std::cout << "Plot with:" << std::endl;
  std::cout << "  python3 ../scripts/allan_variance.py " << outputPath << " "
            << configPath << " --rate " << rate << std::endl;

  return 0;
}
