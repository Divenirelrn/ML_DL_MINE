cmake_minimum_required(VERSION 3.10)
project(torch_demo)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${torch_CXX_FLAGS}")

message("${CMAKE_CURRENT_LIST_DIR}-----------${CMAKE_BINARY_DIR}")
option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

add_executable(main main.cpp)
message("${TORCH_LIBRARIES}--------------")
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 14)
