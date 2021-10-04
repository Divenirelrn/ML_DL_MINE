//
// Created by YH-Mac on 2020/9/11.
//
#include "silent_live_ir.h"
#include "helper.h"
#include "iostream"
#include "opencv2/opencv.hpp"
#include <sstream>
#include <string>
using namespace std;
void print_feature(const std::string &filename, std::vector<float> &feature) {
  int feature_len = feature.size();
  std::cout << filename << ":";
  for (int i = 0; i < feature_len; i++) {
    std::cout << feature[i] << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char **argv) {
 if (argc < 3) {
   std::cout << argv[0] << " ";
   std::cout << "model_path pics_dir features_dir " << std::endl;
   return 0;
 }

    string model_dir = argv[1];
    string images_folder_path = argv[2];

  ZipWrapper zipWrapper(model_dir);
  SilentLiveIR slient_ir;
  slient_ir.Reset(zipWrapper);

  if (check_folder(images_folder_path, false)) {
    std::cout << "images_folder_path: " << images_folder_path << std::endl;
  } else {
    std::cout << "images_folder_path folder does not exist." << std::endl;
    return 1;
  }

  // Read folder
  string suffix = "jpg";
  vector<string> file_names;
  vector<string> path_list = readFileListSuffix(
      images_folder_path.c_str(), suffix.c_str(), file_names, false);
  std::cout << "len path list" << path_list.size() << std::endl;
  int show_cout = 3;
  int show_front_index =
      path_list.size() < show_cout ? path_list.size() : show_cout;
  for (int i = 0; i < show_front_index; ++i) {
    cout << path_list[i] << endl;
  }
  Timer timer;
  std::string txt = "";
  for (int idx = 0; idx < path_list.size(); idx++) {
    std::string path_list_idx = path_list[idx];
    cv::Mat image = cv::imread(path_list[idx]);
    float score = slient_ir.Check(image);
      std::cout <<path_list_idx << " " << score << std::endl;
  }


  return 0;
}
