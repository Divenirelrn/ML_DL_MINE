#!/bin/bash

set -e

# for rk1808 aarch64
#GCC_COMPILER=~/opts/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu

# for rk1806 armhf
#GCC_COMPILER=/rv1109/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf

# for rv1109/rv1126 armhf
#GCC_COMPILER=/rv1109/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf

# 
GCC_COMPILER=/rv1109/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf
#GCC_COMPILER=/usr/bin/

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# build rockx
if [ ! -d "build" ];then
    mkdir build
else:
    rm -r build
    mkdir build
fi

BUILD_DIR=${ROOT_PWD}/build

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
echo ${GCC_COMPILER}gcc
/rv1109/cmake/bin/cmake .. \
    -DCMAKE_C_COMPILER=${GCC_COMPILER}-gcc \
    -DCMAKE_CXX_COMPILER=${GCC_COMPILER}-g++ \
    -DRKNN_API_PATH=/rv1109/nz_face_lite_rknn_v3/librknn_api
make -j40
#make install
#cd -
#cp -r ./test_images install/artifacts/test_images
