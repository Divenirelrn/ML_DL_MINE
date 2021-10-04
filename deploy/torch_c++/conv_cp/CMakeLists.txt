cmake_minimum_required(VERSION 3.10)
project(convnet)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_executable(conv)
target_sources(conv PRIVATE src/main.cpp
					src/convnet.cpp
					include/convnet.h
)
target_include_directories(conv PRIVATE include)
target_link_libraries(conv ${TORCH_LIBRARIES})
set_target_properties(conv PROPERTIES 
				CXX_STANDARD 14
				CXX_STANDARD_REQUIRED YES
)
