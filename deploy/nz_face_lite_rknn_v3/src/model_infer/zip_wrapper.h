//
// Created by 庾金科 on 2020/9/26.
//

#ifndef LMKTRACKING_LIB_ZIP_WRAPPER_H
#define LMKTRACKING_LIB_ZIP_WRAPPER_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>

class ZipWrapper {
public:
  ZipWrapper(std::vector<unsigned char> &data);
  ZipWrapper(const std::string &path);
  std::string ReadFile(const std::string &filename);
  std::vector<char> ReadFileBin(const std::string &filename);
  std::vector<char> ReadFileBinary(const std::string &filename);

private:
  typedef struct Wrapper Wrapper_;
  std::shared_ptr<Wrapper_> wrapper_;
};

#endif // LMKTRACKING_LIB_ZIP_WRAPPER_H
