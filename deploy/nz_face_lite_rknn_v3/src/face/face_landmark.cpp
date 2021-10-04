//======================================================================================
//
//        Copyright (C) 2021 THL A29 Limited, a hangyi company. All rights reserved.
//
//        filename :face_landmark.cpp
//        description : landmark alg on rv1109
//        created by yangjunpei and jackyu at  31/05/2021 18:19:28
//
//======================================================================================

#include "face_landmark.h"

FaceLandmark::FaceLandmark() {}

void FaceLandmark::Reset(ZipWrapper &wrapper) {
  std::string model_config = wrapper.ReadFile("models.yaml");
  Yaml::Node root;
  Yaml::Parse(root, model_config);
  ModelConfig config;
  config.FromYmalNode(root["landmark"]);
  m_infer_.InitRunner(config, &wrapper);
}

void FaceLandmark::Reset(std::string& model_path) {
  
  ZipWrapper yaml_wrapper(model_path + "/libmodels.so");
  std::string model_config = yaml_wrapper.ReadFile("models.yaml");

  Yaml::Node root;
  Yaml::Parse(root, model_config);
  ModelConfig config;
  config.FromYmalNode(root["landmark"]);
  
  std::string modelfile_name = model_path + "/" + config.filename; // "/face_landmark_106_origin.rknn";
  m_infer_.InitRunner(config, modelfile_name);
}



cv::Mat FaceLandmark::ComputeCropMatrix(const cv::Rect2f &rect) {
  float x = rect.x;
  float y = rect.y;
  float w = rect.width;
  float h = rect.height;
  float cx = x + w / 2;
  float cy = y + h / 2;
  float length = std::max(w, h) * 1.0 / 2;
  float x1 = cx - length;
  float y1 = cy - length;
  float x2 = cx + length;
  float y2 = cy + length;
  cv::Rect2f padding_rect(x1, y1, x2 - x1, y2 - y1);
  std::vector<cv::Point2f> rect_pts = Rect2Points(padding_rect);
  rect_pts.erase(rect_pts.end() - 1);
  std::vector<cv::Point2f> dst_pts = {{0, 0}, {112, 0}, {112, 112}};
  cv::Mat m = cv::getAffineTransform(rect_pts, dst_pts);
  return m;
}

void FaceLandmark::Extract(const cv::Mat &image) {
  //int h = image.rows;
  //int w = image.cols;
  //printf("input image height %d ,width %d \n" , h, w);
  cv::Mat image_rgb;
  //cv::resize(image ,image_rgb ,cv::Size(112,112));
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  m_infer_.SetInputData(0, image_rgb);
  m_infer_.RunModel();
  const float *data = m_infer_.GetOutputData(0);
  std::vector<int> output_tensorsize = m_infer_.GetOutputTensorSize(0);
  //for(auto &size:output_tensorsize){
  //    printf("%d " , size);
  //}
  //  printf("\n");

    std::vector<float> res(data, data + 212);
  pts106_.resize(106);
  for (int i = 0; i < 106; i++) {
    pts106_[i].x = static_cast<float>(res[i * 2]*112);
    pts106_[i].y = static_cast<float>(res[i * 2 + 1]*112);
  }
}

void FaceLandmark::Extract(const cv::Mat &image, const cv::Rect2f& rect) {
  cv::Mat affine_land106 = ComputeCropMatrix(rect);
  cv::Mat invert_affine_land106;
  cv::invertAffineTransform(affine_land106, invert_affine_land106);
  cv::Mat crop_img_land106;
  cv::warpAffine(image, crop_img_land106, affine_land106, cv::Size(112,112));

  //cv::imwrite("106_landmarks_putnet.jpg", crop_img_land106);
  
  cv::Mat image_rgb;
  cv::cvtColor(crop_img_land106, image_rgb, cv::COLOR_BGR2RGB);
  m_infer_.SetInputData(0, image_rgb);
  m_infer_.RunModel();
  const float *data = m_infer_.GetOutputData(0);
  std::vector<int> output_tensorsize = m_infer_.GetOutputTensorSize(0);
  //for(auto &size:output_tensorsize){
  //    printf("%d " , size);
  //}
  // printf("\n");

  std::vector<float> res(data, data + 212);
  pts106_.resize(106);
  for (int i = 0; i < 106; i++) {
    pts106_[i].x = (res[i * 2]*112);
    pts106_[i].y = (res[i * 2 + 1]*112);
  }
  sourceimagepts.resize(106);
  sourceimagepts = ApplyTransformToPoints(pts106_, invert_affine_land106);
}

std::vector<cv::Point2f> FaceLandmark::getPoints() const { return pts106_; }

std::vector<cv::Point2f> FaceLandmark::getSourceImagePoints() const { return sourceimagepts; }


