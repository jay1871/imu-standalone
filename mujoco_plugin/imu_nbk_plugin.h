#pragma once

#include <memory>
#include <optional>
#include <string>

#include <mujoco/mujoco.h>

#include "imu_simulator.h"

namespace imu_nbk_plugin {

class ImuNbkSensor {
 public:
  static std::optional<ImuNbkSensor> Create(const mjModel* m, mjData* d,
                                            int instance);
  void Reset(const mjModel* m, int instance);
  void Compute(const mjModel* m, mjData* d, int instance);

  static void RegisterPlugin();

 private:
  ImuNbkSensor(std::unique_ptr<imu_simulator::ImuSimulator> sim,
               int site_id);

  std::unique_ptr<imu_simulator::ImuSimulator> sim_;
  int site_id_;
};

}  // namespace imu_nbk_plugin
