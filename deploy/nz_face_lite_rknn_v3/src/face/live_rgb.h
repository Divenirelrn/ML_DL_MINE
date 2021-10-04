//======================================================================================
//
//        Copyright (C) 2021 THL A29 Limited, a hangyi company. All rights reserved.
//
//        filename :live_rgb.h
//        description : live rgb alg on rv1109
//        created by yangjunpei at  11/08/2021 18:19:28
//
//======================================================================================

#ifndef LIVE_RGB_H
#define LIVE_RGB_H

#include "rknn_infer.h"
#include <opencv2/opencv.hpp>

#define MOVE_MIDDLE_SCREEN (66.66)  // remind the user move to the middle of the screen

class RgbLive {
public:
  RgbLive();
  void Reset(ZipWrapper &wrapper);
  void Reset(std::string& model_path);
  cv::Mat ComputeCropMatrix(const cv::Rect2f& rect, float expand);
  cv::Rect norm_crop_live(const cv::Rect2f& rect, float expand, int img_w, int img_h, int img_size);
  // for crop image
  float Check(const cv::Mat &crop_img);
  
  // for source image, save crop image
  float Check(const cv::Mat &img, const cv::Rect2f& rect, std::string outName);

  // for source image
  float Check(const cv::Mat &img, const cv::Rect2f& rect);

private:
  ModelInferRKNN m_infer_;
};

#endif // LIVE_RGB_H
