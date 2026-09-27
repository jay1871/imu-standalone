#include "imu_nbk_plugin.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "config_loader.h"
#include "imu_simulator.h"

namespace imu_nbk_plugin {

namespace {

int SensorSiteId(const mjModel* m, int instance) {
  for (int i = 0; i < m->nsensor; i++) {
    if (m->sensor_type[i] == mjSENS_PLUGIN &&
        m->sensor_plugin[i] == instance) {
      return m->sensor_objid[i];
    }
  }
  return -1;
}

}  // namespace

ImuNbkSensor::ImuNbkSensor(std::unique_ptr<imu_simulator::ImuSimulator> sim,
                           int site_id)
    : sim_(std::move(sim)), site_id_(site_id) {}

std::optional<ImuNbkSensor> ImuNbkSensor::Create(const mjModel* m, mjData*,
                                                  int instance) {
  const char* config_path = mj_getPluginConfig(m, instance, "config_path");
  if (!config_path || std::strlen(config_path) == 0) {
    std::cerr << "imu_nbk: missing 'config_path' attribute\n";
    return std::nullopt;
  }

  std::string resolved_path = config_path;
  if (!std::filesystem::exists(resolved_path)) {
    std::string filename = std::filesystem::path(config_path).filename().string();
    std::vector<std::string> candidates = {
      std::string("../") + config_path,
      std::string("config/") + filename,
      std::string("../config/") + filename,
      std::string("../../config/") + filename,
    };
    for (const auto& cand : candidates) {
      if (std::filesystem::exists(cand)) {
        resolved_path = cand;
        break;
      }
    }
  }

  int site_id = SensorSiteId(m, instance);
  if (site_id < 0) {
    std::cerr << "imu_nbk: plugin must be attached to a site sensor\n";
    return std::nullopt;
  }

  try {
    auto sim_obj = imu_simulator::createSimulatorFromConfig(resolved_path);
    auto sim = std::make_unique<imu_simulator::ImuSimulator>(std::move(sim_obj));
    sim->setImuSampleTime(m->opt.timestep);
    // MuJoCo's accelerometer already outputs specific force (includes gravity).
    // Disable the internal gravity model to avoid subtracting gravity twice.
    sim->setAccEnableGravity(false);
    return ImuNbkSensor(std::move(sim), site_id);
  } catch (const std::exception& e) {
    std::cerr << "imu_nbk: failed to load config: " << e.what() << "\n";
    return std::nullopt;
  }
}

void ImuNbkSensor::Reset(const mjModel* m, int instance) {
  (void)m;
  (void)instance;
  sim_->resetImuSimulator();
}

void ImuNbkSensor::Compute(const mjModel* m, mjData* d, int instance) {
  // Strategy: Read the true specific force and angular velocity from MuJoCo's
  // built-in accelerometer and gyro sensors (defined before this plugin sensor
  // in the XML, so already computed). These give the physically correct values
  // at the site — including gravity, centripetal, coriolis, etc.
  // Then we apply the NBK error model on top.

  // Find the built-in accelerometer and gyro sensors on the same site
  Eigen::Vector3d true_acc = Eigen::Vector3d::Zero();
  Eigen::Vector3d true_gyro = Eigen::Vector3d::Zero();
  bool found_acc = false, found_gyro = false;

  for (int i = 0; i < m->nsensor; i++) {
    if (m->sensor_objid[i] == site_id_) {
      if (m->sensor_type[i] == mjSENS_ACCELEROMETER) {
        int adr = m->sensor_adr[i];
        true_acc = Eigen::Vector3d(d->sensordata[adr],
                                   d->sensordata[adr + 1],
                                   d->sensordata[adr + 2]);
        found_acc = true;
      } else if (m->sensor_type[i] == mjSENS_GYRO) {
        int adr = m->sensor_adr[i];
        true_gyro = Eigen::Vector3d(d->sensordata[adr],
                                    d->sensordata[adr + 1],
                                    d->sensordata[adr + 2]);
        found_gyro = true;
      }
    }
    if (found_acc && found_gyro) break;
  }

  if (!found_acc || !found_gyro) {
    std::cerr << "imu_nbk: requires accelerometer and gyro sensors on the same "
                 "site (defined before the plugin sensor in XML)\n";
    return;
  }

  // Site rotation matrix — used for body-to-nav quaternion
  const double* site_xmat = d->site_xmat + 9 * site_id_;
  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R_ws(
      site_xmat);
  Eigen::Matrix3d R_mat = R_ws.transpose();
  Eigen::Quaterniond q_b_n(R_mat);

  // Apply error model: true_acc/true_gyro are already in site frame (specific force)
  Eigen::Vector3d acc_meas =
      sim_->generateAccelerationMeasurement(true_acc, q_b_n);
  Eigen::Vector3d gyro_meas =
      sim_->generateGyroscopeMeasurement(true_gyro, q_b_n);

  // Find sensor data offset
  for (int i = 0; i < m->nsensor; i++) {
    if (m->sensor_type[i] == mjSENS_PLUGIN &&
        m->sensor_plugin[i] == instance) {
      int adr = m->sensor_adr[i];
      d->sensordata[adr + 0] = acc_meas(0);
      d->sensordata[adr + 1] = acc_meas(1);
      d->sensordata[adr + 2] = acc_meas(2);
      d->sensordata[adr + 3] = gyro_meas(0);
      d->sensordata[adr + 4] = gyro_meas(1);
      d->sensordata[adr + 5] = gyro_meas(2);
      break;
    }
  }
}

void ImuNbkSensor::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sensor.imu_nbk";
  plugin.capabilityflags |= mjPLUGIN_SENSOR;
  plugin.needstage = mjSTAGE_ACC;

  static const char* attributes[] = {"config_path"};
  plugin.nattribute = 1;
  plugin.attributes = attributes;

  plugin.nstate = +[](const mjModel*, int) -> int { return 12; };

  plugin.nsensordata = +[](const mjModel*, int, int) -> int { return 6; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) -> int {
    auto sensor = ImuNbkSensor::Create(m, d, instance);
    if (!sensor.has_value()) return -1;
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(
        new ImuNbkSensor(std::move(*sensor)));
    return 0;
  };

  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<ImuNbkSensor*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };

  plugin.reset = +[](const mjModel* m, mjtNum*, void* plugin_data,
                     int instance) {
    auto* sensor = reinterpret_cast<ImuNbkSensor*>(plugin_data);
    if (sensor) sensor->Reset(m, instance);
  };

  plugin.compute = +[](const mjModel* m, mjData* d, int instance, int) {
    auto* sensor =
        reinterpret_cast<ImuNbkSensor*>(d->plugin_data[instance]);
    if (sensor) sensor->Compute(m, d, instance);
  };

  mjp_registerPlugin(&plugin);
}

}  // namespace imu_nbk_plugin
