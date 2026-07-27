cmake_minimum_required(VERSION 3.14)
project(imu_simulator VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_APPS "Build application executables" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- Dependencies ---
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)

# --- Core Library (imu_simulator) - depends only on Eigen3 ---
add_library(imu_simulator
  src/imu_simulator.cpp
)

target_include_directories(imu_simulator
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/imu_simulator>
    $<INSTALL_INTERFACE:include/imu_simulator>
)

target_link_libraries(imu_simulator
  PUBLIC Eigen3::Eigen
)

set_target_properties(imu_simulator PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)

# --- Config Loader Library (adds yaml-cpp dependency) ---
add_library(imu_simulator_config
  src/config_loader.cpp
)

target_include_directories(imu_simulator_config
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/imu_simulator>
    $<INSTALL_INTERFACE:include/imu_simulator>
)

target_link_libraries(imu_simulator_config
  PUBLIC imu_simulator yaml-cpp
)

# --- Application executables ---
if(BUILD_APPS)
  add_executable(allan_validation apps/allan_validation.cpp)
  target_link_libraries(allan_validation PRIVATE imu_simulator_config)
endif()

# --- Tests ---
if(BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)

  add_executable(imu_simulator_test
    test/main.cpp
    test/imu_simulator_test.cpp
  )

  target_link_libraries(imu_simulator_test
    PRIVATE imu_simulator GTest::gtest
  )

  add_test(NAME imu_simulator_test COMMAND imu_simulator_test)
endif()

# --- Install ---
include(GNUInstallDirs)

install(TARGETS imu_simulator imu_simulator_config
  EXPORT imu_simulator_targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/imu_simulator
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT imu_simulator_targets
  FILE imu_simulatorTargets.cmake
  NAMESPACE imu_simulator::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/imu_simulator
)
