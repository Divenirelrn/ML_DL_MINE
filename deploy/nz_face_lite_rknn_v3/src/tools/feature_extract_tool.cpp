//
// Created by YH-Mac on 2020/9/11.
//
#include "feature_extract.h"
#include "helper.h"
#include "iostream"
#include "opencv2/opencv.hpp"
#include <sstream>
#include <string>

using namespace std;
void print_feature(const std::string &filename, std::vector<float> &feature) {
  int feature_len = feature.size();
  std::cout << filename << ":";
  for (int i = 0; i < 10; i++) {
    std::cout << feature[i] << ",";
  }
  std::cout << std::endl;
}

int main(int argc, char **argv) {
 if (argc < 4) {
   std::cout << argv[0] << " ";
   std::cout << "model_path pics_dir features_dir " << std::endl;
   return 0;
 }

//  string model_dir = "/Users/yujinke/Downloads/rknn_face_embed/model/model.zip";
//  string images_folder_path = "/Users/yujinke/Desktop/align_id/aligned";
//  string features_folder_path = "/Users/yujinke/Downloads/rknn_face_embed/build/features";
    string model_dir = argv[1];
    string images_folder_path = argv[2];
    string features_folder_path = argv[3];
  FILE* feat = fopen(features_folder_path.c_str(), "w");
  
  ZipWrapper zipWrapper(model_dir);
  FeatureExtract featureExtract;
  featureExtract.Reset(zipWrapper);
  std::cout << "load features model sucess!" << std::endl;
  if (check_folder(images_folder_path, false)) {
    std::cout << "images_folder_path: " << images_folder_path << std::endl;
  } else {
    std::cout << "images_folder_path folder does not exist." << std::endl;
    return 1;
  }
  //check_folder(features_folder_path, true);
  // Read folder
  string suffix = "jpg";
  vector<string> file_names;
  vector<string> path_list = readFileListSuffix(
      images_folder_path.c_str(), suffix.c_str(), file_names, false);
  std::cout << "len path list: " << path_list.size() << std::endl;
  int show_cout = 3;
  int show_front_index =
      path_list.size() < show_cout ? path_list.size() : show_cout;
  for (int i = 0; i < show_front_index; ++i) {
    cout << path_list[i] << endl;
  }
  cout << "....." << endl;
  cout << "Search for ." + suffix + " file: " << path_list.size() << endl;
  Timer timer;
  std::string txt = "";
  for (int idx = 0; idx < path_list.size(); idx++) {
    std::string path_list_idx = path_list[idx];
    std::cout << path_list_idx << std::endl;
    cv::Mat image = cv::imread(path_list[idx]);
    std::cout<<"image size "<<image.rows << ","<<image.cols<<std::endl;
    //string feat_txt_path =
    //    features_folder_path + "/" + file_names[idx] + ".txt";
    //std::cout << path_list_idx << "->" << feat_txt_path << std::endl;
    std::vector<float> feature_v = featureExtract.Extract(image);
    //print_feature(path_list_idx, feature_v);
    //extract_feat_to_txt(feat_txt_path, feature_v);
    fprintf(feat, "%s,", file_names[idx].c_str());
    for (int i=0; i<feature_v.size(); i++){
        fprintf(feat,"%f,", feature_v[i]);  
    }
    fprintf(feat,"\n");
    //return 0;
  }
  fclose(feat);
  std::cout << "Feature extraction successful" << path_list.size() << endl;
  return 0;
}
