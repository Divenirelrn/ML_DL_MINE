//======================================================================================
//
//        Copyright (C) 2021 THL A29 Limited, a hangyi company. All rights reserved.
//
//        filename :rgb_live.cpp
//        description : rgb live alg on rv1109
//        created by yangjunpei at  11/08/2021 18:19:28
//
//======================================================================================

#include "live_rgb.h"
#include "utils.h"
#include <utility>

RgbLive::RgbLive() {}

void RgbLive::Reset(ZipWrapper &wrapper) {
  std::string model_config = wrapper.ReadFile("models.yaml");
  Yaml::Node root;
  Yaml::Parse(root, model_config);
  ModelConfig config;
  config.FromYmalNode(root["rgb_live"]);
  m_infer_.InitRunner(config, &wrapper);
}

void RgbLive::Reset(std::string& model_path) {
  ZipWrapper yaml_wrapper(model_path + "/libmodels.so");
  std::string model_config = yaml_wrapper.ReadFile("models.yaml");
  
  Yaml::Node root;
  Yaml::Parse(root, model_config);
  ModelConfig config;
  config.FromYmalNode(root["rgb_live"]);

  std::string modelfile_name = model_path + "/" + config.filename; //"/liveness_ir.rknn";
  m_infer_.InitRunner(config, modelfile_name);
}

cv::Mat RgbLive::ComputeCropMatrix(const cv::Rect2f &rect, float expand) {
  float x = rect.x;
  float y = rect.y;
  float w = rect.width;
  float h = rect.height;
  float cx = x + w / 2;
  float cy = y + h / 2;
  float length = std::max(w, h) * expand / 2;
  float x1 = cx - length;
  float y1 = cy - length;
  float x2 = cx + length;
  float y2 = cy + length;
  cv::Rect2f padding_rect(x1, y1, x2 - x1, y2 - y1);
  std::vector<cv::Point2f> rect_pts = Rect2Points(padding_rect);
  rect_pts.erase(rect_pts.end() - 1);
  std::vector<cv::Point2f> dst_pts = {{0, 0}, {80, 0}, {80, 80}};
  cv::Mat m = cv::getAffineTransform(rect_pts, dst_pts);
  return m;
}

float RgbLive::Check(const cv::Mat &crop_image) {
  assert(crop_image.rows == 80);
  assert(crop_image.cols == 80);
  m_infer_.SetInputData(0, crop_image);
  m_infer_.RunModel();
  const float *output = m_infer_.GetOutputData(0);
  std::vector<int> out_tensor_size = m_infer_.GetOutputTensorSize(0);
  int sum = 1;
  for (int i = 0; i < out_tensor_size.size(); i++) {
    sum *= out_tensor_size[i];
  }
  std::vector<float> res(output, output + sum);
  // printf("rgblive output, %f, %f, %f\n", res[0], res[1], res[2]);
  float live_score = res[1];
  return live_score;
}

int q_max(int x, int y, int z, int h) {
  int ans=0;
  ans = x>y?x:y;
  ans = ans>z?ans:z;
  ans = ans>h?ans:h;
  return ans;
}

cv::Rect RgbLive::norm_crop_live(const cv::Rect2f& rect, float expand, int img_w, int img_h, int img_size) {
  float x  = rect.x;
  float y  = rect.y;
  float w  = rect.width;
  float h  = rect.height;
  float cx = x + w / 2;
  float cy = y + h / 2;
  float bbox_size     = std::min(h, w);
  float new_bbox_size = bbox_size * expand;
  float new_bbox_half = new_bbox_size * 0.5;

  int x1   = int(cx - new_bbox_half);
  int x2   = int(cx + new_bbox_half);
  int y1   = int(cy - new_bbox_half);
  int y2   = int(cy + new_bbox_half);
  int max_oob     = q_max(0-x1, 0-y1, x2-img_w, y2-img_h);
  float oob_ratio = float(max_oob) / new_bbox_size;

  cv::Rect ans(1, 1, 1, 1);
  if (oob_ratio > 0.2) {
    return ans;
  } else {
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(img_w, x2);
    y2 = std::min(img_h, y2);
    ans.x = x1;
    ans.y = y1;
    ans.width  = x2-x1;
    ans.height = y2-y1;
    return ans;
  }
}

 // rgb save image
float RgbLive::Check(const cv::Mat &img, const cv::Rect2f& rect, std::string outName) {

  cv::Rect crop_rect = norm_crop_live(rect, 2.7, img.cols, img.rows, 80);
  if (crop_rect.x == 1 && crop_rect.y == 1 && crop_rect.width == 1 && crop_rect.height == 1) {
    // return MOVE_MIDDLE_SCREEN;
    return -2;
  }

  cv::Mat resize_img;
  cv::Mat nimg;
  cv::Mat ROI(img, crop_rect);
  ROI.copyTo(nimg);
  cv::resize(nimg, resize_img, cv::Size(80, 80));
  
  // cv::imwrite("norm_crop_live.png", resize_img);
  // std::string outName = "../rgb0812tzp_rv_det_crop/" + filename;
  if (outName.length() > 5) {
    cv::imwrite(outName, resize_img);
  }

  assert(resize_img.rows == 80);
  assert(resize_img.cols == 80);
  m_infer_.SetInputData(0, resize_img);
  m_infer_.RunModel();
  const float *output = m_infer_.GetOutputData(0);
  std::vector<int> out_tensor_size = m_infer_.GetOutputTensorSize(0);
  int sum = 1;
  for (int i = 0; i < out_tensor_size.size(); i++) {
    sum *= out_tensor_size[i];
  }
  std::vector<float> res(output, output + sum);
  
  // printf("rgblive output, %f, %f, %f\n", res[0], res[1], res[2]);
  float live_score = res[1];
  return live_score;
}

// rgb
float RgbLive::Check(const cv::Mat &img, const cv::Rect2f& rect) {

  cv::Rect crop_rect = norm_crop_live(rect, 2.7, img.cols, img.rows, 80);
  if (crop_rect.x == 1 && crop_rect.y == 1 && crop_rect.width == 1 && crop_rect.height == 1) {
    return MOVE_MIDDLE_SCREEN;
  }

  cv::Mat resize_img;
  cv::Mat nimg;
  cv::Mat ROI(img, crop_rect);
  ROI.copyTo(nimg);
  cv::resize(nimg, resize_img, cv::Size(80, 80));
  
  // cv::imwrite("norm_crop_live.png", resize_img);
  assert(resize_img.rows == 80);
  assert(resize_img.cols == 80);
  m_infer_.SetInputData(0, resize_img);
  m_infer_.RunModel();
  const float *output = m_infer_.GetOutputData(0);
  std::vector<int> out_tensor_size = m_infer_.GetOutputTensorSize(0);
  int sum = 1;
  for (int i = 0; i < out_tensor_size.size(); i++) {
    sum *= out_tensor_size[i];
  }
  std::vector<float> res(output, output + sum);
  
  // printf("rgblive output, %f, %f, %f\n", res[0], res[1], res[2]);
  float live_score = res[1];
  return live_score;
}