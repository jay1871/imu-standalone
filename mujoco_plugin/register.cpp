#include <mujoco/mjplugin.h>

#include "imu_nbk_plugin.h"

mjPLUGIN_LIB_INIT(imu_nbk) {
  imu_nbk_plugin::ImuNbkSensor::RegisterPlugin();
}
