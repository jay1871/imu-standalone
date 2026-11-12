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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "config_loader.h"

struct AdevResult {
  std::vector<double> taus;
  std::vector<double> acc_adev[3];
  std::vector<double> gyro_adev[3];
};

std::vector<double> computeOctaveTaus(long totalSamples, double dt) {
  std::vector<double> taus;
  if (totalSamples <= 0 || dt <= 0.0) {
    return taus;
  }
  double tau_min = dt;
  double tau_max = static_cast<double>(totalSamples) * dt / 3.0;
  double tau = tau_min;
  while (tau <= tau_max) {
    taus.push_back(tau);
    tau *= 2.0;
  }
  if (!taus.empty() && taus.back() < 10000.0 && tau_max >= 10000.0) {
    taus.push_back(10000.0);
  }
  return taus;
}

std::vector<double> computeOverlappingAdev(const std::vector<long double>& data,
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

    long double sum = 0.0L;
    long count = 0;
    for (long i = 0; i + 2 * m < N; i++) {
      long double diff = data[i + 2 * m] - 2.0L * data[i + m] + data[i];
      sum += diff * diff;
      count++;
    }

    if (count > 0) {
      long double tau_val = static_cast<long double>(m) * static_cast<long double>(dt);
      adev_out[ti] = static_cast<double>(
          std::sqrt(sum / (2.0L * tau_val * tau_val * static_cast<long double>(count))));
    }
  }
  return adev_out;
}

#ifdef ENABLE_DUT_PARAMETER_SPREAD
imu_simulator::ImuSimParams sampleDutParams(
    const imu_simulator::ImuSimParams& base, int seed_offset,
    unsigned int base_seed) {
  imu_simulator::ImuSimParams p = base;
  if (seed_offset <= 0) {
    return p;
  }
  // Use a dedicated RNG stream for virtual DUT part-to-part parameter sampling
  std::mt19937 rng(base_seed + 100000U + static_cast<unsigned int>(seed_offset));
  std::normal_distribution<double> norm(0.0, 1.0);

  auto samplePositiveZ = [&](double mean, double std_val, double z) -> double {
    if (mean <= 0.0 || std_val <= 0.0) return mean;
    // Log-normal distribution matching specified mean and std (guarantees positive parameters)
    double cv2 = (std_val / mean) * (std_val / mean);
    double sigma_ln = std::sqrt(std::log(1.0 + cv2));
    double mu_ln = std::log(mean) - 0.5 * sigma_ln * sigma_ln;
    return std::exp(mu_ln + sigma_ln * z);
  };

  const double rho_bk = 0.85;
  const double rho_bk_comp = std::sqrt(1.0 - rho_bk * rho_bk);

  for (int ax = 0; ax < 3; ax++) {
    double z_n = norm(rng);
    double z_b = norm(rng);
    double z_k = rho_bk * z_b + rho_bk_comp * norm(rng);
    double z_tc = norm(rng);
    p.N(ax) = samplePositiveZ(base.N(ax), base.N_std(ax), z_n);
    p.B(ax) = samplePositiveZ(base.B(ax), base.B_std(ax), z_b);
    p.K(ax) = samplePositiveZ(base.K(ax), base.K_std(ax), z_k);
    p.corrTime(ax) = samplePositiveZ(base.corrTime(ax), base.corrTime_std(ax), z_tc);
    if (base.B(ax) > 0.0) {
      double b_scale = p.B(ax) / base.B(ax);
      p.B_poles.row(ax) = base.B_poles.row(ax) * b_scale;
    }
    if (base.corrTime(ax) > 0.0) {
      double tc_scale = p.corrTime(ax) / base.corrTime(ax);
      p.corrTime_poles.row(ax) = base.corrTime_poles.row(ax) * tc_scale;
    }
  }
  return p;
}
#endif

AdevResult runSingleSeed(imu_simulator::SimulatorConfig& config, double dt,
                          long totalSamples, int seed, int seedIndex,
                          const std::vector<double>& taus,
                          bool enableDutSpread) {
#ifdef ENABLE_DUT_PARAMETER_SPREAD
  auto accParams = enableDutSpread
                       ? sampleDutParams(config.accSimParams, seedIndex,
                                         static_cast<unsigned int>(config.seed))
                       : config.accSimParams;
  auto gyroParams = enableDutSpread
                        ? sampleDutParams(config.gyroSimParams, seedIndex,
                                          static_cast<unsigned int>(config.seed))
                        : config.gyroSimParams;
#else
  (void)seedIndex;
  (void)enableDutSpread;
  const auto& accParams = config.accSimParams;
  const auto& gyroParams = config.gyroSimParams;
#endif

  imu_simulator::ImuSimulator sim(accParams, gyroParams,
                                   config.enableSettings, config.geoPositionLlh,
                                   config.geoVelocityNed, dt, seed);
  sim.resetImuSimulator();

  Eigen::Vector3d a_true = Eigen::Vector3d::Zero();
  Eigen::Vector3d w_true = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_identity(1.0, 0.0, 0.0, 0.0);

  // Phase-integrated output (N + 1 phase points for N frequency samples)
  std::vector<long double> acc_data[3];
  std::vector<long double> gyro_data[3];
  for (int ax = 0; ax < 3; ax++) {
    acc_data[ax].reserve(totalSamples + 1);
    gyro_data[ax].reserve(totalSamples + 1);
    acc_data[ax].push_back(0.0L);
    gyro_data[ax].push_back(0.0L);
  }

  const long double dt_ld = static_cast<long double>(dt);

  // Generate time series in memory
  for (long i = 0; i < totalSamples; i++) {
    Eigen::Vector3d acc = sim.generateAccelerationMeasurement(a_true, q_identity);
    Eigen::Vector3d gyro = sim.generateGyroscopeMeasurement(w_true, q_identity);

    // Store phase (integrated) for OADEV — cumulative sum × dt in 80-bit extended precision
    for (int ax = 0; ax < 3; ax++) {
      acc_data[ax].push_back(acc_data[ax].back() + static_cast<long double>(acc(ax)) * dt_ld);
      gyro_data[ax].push_back(gyro_data[ax].back() + static_cast<long double>(gyro(ax)) * dt_ld);
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
  bool forceNoSpread = false;

  std::vector<std::string> posArgs;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--no-spread") {
      forceNoSpread = true;
    } else {
      posArgs.push_back(arg);
    }
  }

  if (posArgs.size() > 0) configPath = posArgs[0];
  if (posArgs.size() > 1) outputPath = posArgs[1];
  if (posArgs.size() > 2) durationHours = std::atof(posArgs[2].c_str());
  if (posArgs.size() > 3) sampleRateHz = std::atof(posArgs[3].c_str());
  if (posArgs.size() > 4) numSeeds = std::atoi(posArgs[4].c_str());

  if (durationHours <= 0.0) {
    std::cerr << "Error: duration_hours must be > 0\n";
    return 1;
  }
  if (numSeeds < 1) {
    std::cerr << "Error: num_seeds must be >= 1\n";
    return 1;
  }

  std::cout << "Loading config: " << configPath << std::endl;

  imu_simulator::SimulatorConfig config =
      imu_simulator::loadConfig(configPath);

  if (sampleRateHz > 0.0) {
    config.sampleTime = 1.0 / sampleRateHz;
  }

  double dt = config.sampleTime;
  if (dt <= 0.0) {
    std::cerr << "Error: sample_time must be > 0\n";
    return 1;
  }
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
  double memPerRun = (static_cast<double>(totalSamples + 1) * 6.0 *
                      static_cast<double>(sizeof(long double))) /
                     (1024.0 * 1024.0);

  bool enableDutSpread = (numSeeds > 1) && !forceNoSpread;

  std::cout << "Allan Deviation Validation (in-memory)" << std::endl;
  std::cout << "  Config:       " << configPath << std::endl;
  std::cout << "  Output:       " << outputPath << std::endl;
  std::cout << "  Duration:     " << durationHours << " hours" << std::endl;
  std::cout << "  Sample rate:  " << rate << " Hz" << std::endl;
  std::cout << "  Total samples:" << totalSamples << std::endl;
  std::cout << "  RAM per run:  " << static_cast<int>(memPerRun) << " MB" << std::endl;
  std::cout << "  Seeds:        " << numSeeds << std::endl;
#ifdef ENABLE_DUT_PARAMETER_SPREAD
  std::cout << "  DUT Spread:   "
            << (enableDutSpread ? "ENABLED (multi-DUT population)"
                                : "DISABLED (single nominal DUT)")
            << std::endl;
#else
  std::cout << "  DUT Spread:   DISABLED (compiled with ENABLE_DUT_PARAMETER_SPREAD=OFF)" << std::endl;
#endif
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

  std::vector<AdevResult> allResults(numSeeds);
  std::atomic<int> completedSeeds{0};

#ifdef _OPENMP
  int maxThreads = omp_get_max_threads();
  // Cap default concurrency based on RAM per seed (~24 GB budget) unless OMP_NUM_THREADS is set
  int safeThreads = std::max(1, static_cast<int>(24000.0 / std::max(1.0, memPerRun)));
  int numThreads = std::min({numSeeds, maxThreads, safeThreads});
  if (std::getenv("OMP_NUM_THREADS") != nullptr) {
    numThreads = std::min(numSeeds, maxThreads);
  }
  if (numSeeds > 1) {
    std::cout << "  Threads:      " << numThreads << " (OpenMP)" << std::endl;
  }
#pragma omp parallel for schedule(dynamic, 1) num_threads(numThreads)
#endif
  for (int s = 0; s < numSeeds; s++) {
    int seed = config.seed + s;
    auto runStart = std::chrono::steady_clock::now();

    if (numSeeds == 1) {
      std::cout << "\r  Seed 1/1 (seed=" << seed << ") — simulating..." << std::flush;
    }

    allResults[s] =
        runSingleSeed(config, dt, totalSamples, seed, s, taus, enableDutSpread);

    auto runEnd = std::chrono::steady_clock::now();
    double runElapsed = std::chrono::duration<double>(runEnd - runStart).count();
    double totalElapsed = std::chrono::duration<double>(runEnd - totalStart).count();
    int done = ++completedSeeds;
    double eta = (totalElapsed / done) * (numSeeds - done);

#ifdef _OPENMP
#pragma omp critical
#endif
    {
      std::cout << "\r  Seed " << done << "/" << numSeeds
                << " done (" << static_cast<int>(runElapsed) << "s)"
                << " | Total: " << static_cast<int>(totalElapsed) << "s"
                << " | ETA: " << static_cast<int>(eta) << "s     " << std::flush;
    }
  }
  std::cout << std::endl;

  // Deterministic Welford reduction in seed order (0 .. numSeeds - 1)
  for (int s = 0; s < numSeeds; s++) {
    const auto& result = allResults[s];
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
  }

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
        double std_val =
            (numSeeds > 1)
                ? std::sqrt(std::max(0.0, acc_m2[ax][ti] / (numSeeds - 1)))
                : 0.0;
        outFile << "," << acc_mean[ax][ti] << "," << std_val;
      }
      for (int ax = 0; ax < 3; ax++) {
        double std_val =
            (numSeeds > 1)
                ? std::sqrt(std::max(0.0, gyro_m2[ax][ti] / (numSeeds - 1)))
                : 0.0;
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
