//======================================================================================
//
//        Copyright (C) 2021 THL A29 Limited, a hangyi company. All rights reserved.
//
//        filename :face_quality.cpp
//        description : quality alg on rv1109
//        created by yangjunpei and jackyu at  31/05/2021 18:19:28
//
//======================================================================================

#include "face_quality.h"

#define WIDTH 96
#define HEIGHT 96

FaceQuality::FaceQuality() {}

void FaceQuality::Reset(ZipWrapper &wrapper) {
  std::string model_config = wrapper.ReadFile("models.yaml");
  Yaml::Node root;
  Yaml::Parse(root, model_config);
  ModelConfig config;
  config.FromYmalNode(root["posem"]);
  m_infer_.InitRunner(config, &wrapper);
}

void FaceQuality::Reset(std::string& model_path) {
  ZipWrapper yaml_wrapper(model_path + "/libmodels.so");
  std::string model_config = yaml_wrapper.ReadFile("models.yaml");
  Yaml::Node root;
  Yaml::Parse(root, model_config);
  ModelConfig config;
  config.FromYmalNode(root["posem"]);

  std::string modelfile_name = model_path + "/" + config.filename; //"/posem.rknn";
  m_infer_.InitRunner(config, modelfile_name);
}

cv::Mat FaceQuality::ComputeCropMatrix(const cv::Rect2f &rect) {
  float x = rect.x;
  float y = rect.y;
  float w = rect.width;
  float h = rect.height;
  float cx = x + w / 2;
  float cy = y + h / 2;
  float length = std::max(w, h) * 1.5 / 2;
  float x1 = cx - length;
  float y1 = cy - length;
  float x2 = cx + length;
  float y2 = cy + length;
  cv::Rect2f padding_rect(x1, y1, x2 - x1, y2 - y1);
  std::vector<cv::Point2f> rect_pts = Rect2Points(padding_rect);
  rect_pts.erase(rect_pts.end() - 1);
  std::vector<cv::Point2f> dst_pts = {{0, 0}, {96, 0}, {96, 96}};
  cv::Mat m = cv::getAffineTransform(rect_pts, dst_pts);
  return m;
}

void FaceQuality::Extract(const cv::Mat &image) {
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  m_infer_.SetInputData(0, image_rgb);
  m_infer_.RunModel();
  const float *data = m_infer_.GetOutputData(0);
  std::vector<float> res(data, data + 18);
  pitch_ = res[0] * 90;
  yaw_ = res[1] * 90;
  roll_ = res[2] * 90;
  std::vector<float> quality(res.begin() + 13, res.end());
  //printf("quality size:%d\n", quality.size());
  quality_.push_back((quality[0]+quality[1])/2);
  quality_.push_back(quality[2]);
  quality_.push_back((quality[3]+quality[4])/2);

  //quality_ = quality;
  std::vector<float> face_pts5(res.begin() + 3, res.begin() + 13);
  pts5_.resize(5);
  for (int i = 0; i < 5; i++) {
    pts5_[i].x = (face_pts5[i * 2] + 1) * (96 / 2);
    pts5_[i].y = (face_pts5[i * 2 + 1] + 1) * (96 / 2);
  }
}

void FaceQuality::Extract(const cv::Mat &image, const cv::Rect2f &rect) {
  cv::Mat affine = ComputeCropMatrix(rect);
  cv::Mat invert_affine;
  cv::invertAffineTransform(affine, invert_affine);
  cv::Mat crop_img;
  cv::warpAffine(image, crop_img, affine, cv::Size(96,96));

  //cv::imwrite("quality_crop_putnet.jpg", crop_img);
  
  cv::Mat image_rgb;
  cv::cvtColor(crop_img, image_rgb, cv::COLOR_BGR2RGB);
  m_infer_.SetInputData(0, image_rgb);
  m_infer_.RunModel();
  const float *data = m_infer_.GetOutputData(0);
  std::vector<float> res(data, data + 18);
  pitch_ = res[0] * 90;
  yaw_ = res[1] * 90;
  roll_ = res[2] * 90;
  std::vector<float> quality(res.begin() + 13, res.end());
  //printf("quality size:%d\n", quality.size());
  quality_.clear();
  quality.resize(3);
  quality_.push_back((quality[0]+quality[1])/2);
  quality_.push_back(quality[2]);
  quality_.push_back((quality[3]+quality[4])/2);

  //quality_ = quality;
  std::vector<float> face_pts5(res.begin() + 3, res.begin() + 13);
  pts5_.clear();
  pts5_.resize(5);
  for (int i = 0; i < 5; i++) {
    pts5_[i].x = (face_pts5[i * 2] + 1) * (96 / 2);
    pts5_[i].y = (face_pts5[i * 2 + 1] + 1) * (96 / 2);
  }
  sourceimagepts.clear();
  sourceimagepts.resize(5);
  sourceimagepts = ApplyTransformToPoints(pts5_, invert_affine);
  return;
}

std::vector<float> FaceQuality::getQuality() const { return quality_; }

float FaceQuality::getYaw() const { return yaw_; }

float FaceQuality::getRoll() const { return roll_; }

float FaceQuality::getPitch() const { return pitch_; }

std::vector<cv::Point2f> FaceQuality::getPoints() const { return pts5_; }

std::vector<cv::Point2f> FaceQuality::getSourceImagePoints() const { return sourceimagepts; }
