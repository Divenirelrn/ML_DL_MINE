//
// Created by YH-Mac on 2020/9/11.
//

#include "zip_wrapper.h"
#include "zip_file.hpp"
#include <utility>

typedef struct Wrapper {
  miniz_cpp::zip_file zipFile;
} Wrapper;

ZipWrapper::ZipWrapper(std::vector<unsigned char> &data_in) {
  wrapper_.reset(new Wrapper);
  wrapper_->zipFile.load(data_in);
}

ZipWrapper::ZipWrapper(const std::string &path) {
  wrapper_.reset(new Wrapper);
  wrapper_->zipFile.load(path);
}

std::string ZipWrapper::ReadFile(const std::string &filename) {
  assert(wrapper_->zipFile.has_file(filename));
  return wrapper_->zipFile.read(filename);
}

std::vector<char> ZipWrapper::ReadFileBin(const std::string &filename) {
  std::string data = wrapper_->zipFile.read(filename);
  std::vector<char> ok(data.data(), data.data() + data.size());
  return ok;
}

std::vector<char> ZipWrapper::ReadFileBinary(const std::string &filename) {
  return wrapper_->zipFile.readbin(filename);
  // return ok;
}
