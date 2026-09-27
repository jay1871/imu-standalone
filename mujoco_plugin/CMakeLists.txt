# MuJoCo can be provided two ways:
#   1. CMake config (from source install): -Dmujoco_DIR=/path/to/lib/cmake/mujoco
#   2. Prebuilt release directory:         -DMUJOCO_ROOT=/path/to/mujoco-3.x.x
#
# The prebuilt release doesn't ship mujocoConfig.cmake, so we handle both.

find_package(mujoco QUIET)

if(NOT mujoco_FOUND)
  # Fall back to manual discovery from MUJOCO_ROOT
  if(NOT MUJOCO_ROOT)
    message(FATAL_ERROR
      "MuJoCo not found. Provide one of:\n"
      "  -Dmujoco_DIR=/path/to/lib/cmake/mujoco  (source install)\n"
      "  -DMUJOCO_ROOT=/path/to/mujoco-3.x.x     (prebuilt release)\n")
  endif()

  # Resolve to absolute path
  get_filename_component(MUJOCO_ROOT "${MUJOCO_ROOT}" ABSOLUTE)

  # Create imported target manually
  # Use INTERFACE library + manual link flags to avoid IMPORTED_LOCATION
  # path-with-spaces breakage in the linker command line.
  add_library(mujoco::mujoco INTERFACE IMPORTED)

  if(EXISTS "${MUJOCO_ROOT}/lib/libmujoco.so")
    set(MUJOCO_LIB_FILE "${MUJOCO_ROOT}/lib/libmujoco.so")
  elseif(EXISTS "${MUJOCO_ROOT}/lib/libmujoco.dylib")
    set(MUJOCO_LIB_FILE "${MUJOCO_ROOT}/lib/libmujoco.dylib")
  elseif(EXISTS "${MUJOCO_ROOT}/bin/mujoco.dll")
    set(MUJOCO_LIB_FILE "${MUJOCO_ROOT}/bin/mujoco.dll")
  else()
    message(FATAL_ERROR "Could not find libmujoco in ${MUJOCO_ROOT}/lib")
  endif()

  set_target_properties(mujoco::mujoco PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_ROOT}/include"
    INTERFACE_LINK_LIBRARIES "${MUJOCO_LIB_FILE}"
  )

  message(STATUS "Using MuJoCo prebuilt release: ${MUJOCO_ROOT}")
  message(STATUS "  Library: ${MUJOCO_LIB_FILE}")
  message(STATUS "  Headers: ${MUJOCO_ROOT}/include")
endif()

add_library(imu_nbk_plugin SHARED
  imu_nbk_plugin.cpp
  register.cpp
)

target_link_libraries(imu_nbk_plugin
  PRIVATE
    imu_simulator
    imu_simulator_config
    mujoco::mujoco
    Eigen3::Eigen
)

target_include_directories(imu_nbk_plugin
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include/imu_simulator
)

set_target_properties(imu_nbk_plugin PROPERTIES
  PREFIX ""
  OUTPUT_NAME "imu_nbk_plugin"
)
