//======================================================================================
//
//        Copyright (C) 2021 THL A29 Limited, a hangyi company. All rights reserved.
//
//        filename :face_quality.h
//        description : quality alg on rv1109
//        created by yangjunpei and jackyu at  31/05/2021 18:19:28
//
//======================================================================================

#ifndef FACE_QUALITY_H
#define FACE_QUALITY_H
#include "rknn_infer.h"
#include "utils.h"
#include <opencv2/opencv.hpp>

class FaceQuality {
public:
  FaceQuality();
  void Reset(ZipWrapper& wrapper);
  void Reset(std::string& model_path);
  
  // for crop image
  void Extract(const cv::Mat& image);

  // for source image and detect boxes
  void Extract(const cv::Mat& image, const cv::Rect2f& rect);
  cv::Mat ComputeCropMatrix(const cv::Rect2f& rect);
  std::vector<float> getQuality() const;
  float getYaw() const;
  float getRoll() const;
  float getPitch() const;
  std::vector<cv::Point2f> getPoints() const;
  std::vector<cv::Point2f> getSourceImagePoints() const;

private:
  ModelInferRKNN m_infer_;
  std::vector<float> quality_;
  std::vector<cv::Point2f> pts5_;           // for (96,96) size
  std::vector<cv::Point2f> sourceimagepts;  // for source images rows and cols size, used for aligned and extract features
  float yaw_;
  float roll_;
  float pitch_;
};

#endif // FACE_QUALITY_H
