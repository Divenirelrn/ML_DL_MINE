cmake_minimum_required(VERSION 3.10.0)
project(rknn_mobilenet)
set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(STATUS "64bit")
	set(LIB_ARCH lib64)
else()
	message(STATUS "32bit")
	set(LIB_ARCH lib)
endif()

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/librknn_api)
include_directories(${RKNN_API_PATH}/include)
include_directories(3rdparty/libnpy)
include_directories(3rdparty/mini-yaml/yaml)
include_directories(3rdparty/miniz)
include_directories(src/fatigue)
include_directories(src/model_infer)

if(APPLE)
	set(SYS_NAME darwin)
endif()

# opencv
if(${CMAKE_C_COMPILER} MATCHES "arm")
    message("ARM ARCH")
	set(SYS_NAME arm)
	if(LIB_ARCH STREQUAL "lib")
		set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
		message(${OpenCV_DIR})
		message("streequal lib")
	else()
		set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
		message("else here")
	endif()
else()
	set(OpenCV_DIR /Users/yujinke/Downloads/opencv4.5/opencv-4.5.0/build)
endif()

link_directories(${RKNN_API_PATH}/lib/${SYS_NAME}/${LIB_ARCH})
message("${RKNN_API_PATH}/lib/${SYS_NAME}/${LIB_ARCH}")

find_package(OpenCV REQUIRED)
set(CMAKE_INSTALL_RPATH "lib")

#lib
set(MINI_YAML ${CMAKE_SOURCE_DIR}/3rdparty/mini-yaml/yaml/Yaml.cpp)
set(SRC src/fatigue/mobilenet.cpp src/model_infer/zip_wrapper.cpp src/model_infer/rknn_infer.cpp)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
add_library(fatigue_rknn SHARED ${SRC} ${MINI_YAML})
target_link_libraries(fatigue_rknn rknn_api ${OpenCV_LIBS})

#executable
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
add_executable(fatigue_api src/fatigue_recog.cpp)
target_link_libraries(fatigue_api rknn_api ${OpenCV_LIBS} fatigue_rknn)

