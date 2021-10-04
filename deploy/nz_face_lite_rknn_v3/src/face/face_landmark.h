//======================================================================================
//
//        Copyright (C) 2021 THL A29 Limited, a hangyi company. All rights reserved.
//
//        filename :face_landmark.h
//        description : landmark alg on rv1109
//        created by yangjunpei and jackyu at  31/05/2021 18:19:28
//
//======================================================================================

#ifndef FACE_LANDMARK_H
#define FACE_LANDMARK_H

#include "rknn_infer.h"
#include "utils.h"
#include <opencv2/opencv.hpp>

class FaceLandmark {
public:
  FaceLandmark();
  void Reset(ZipWrapper &wrapper);
  void Reset(std::string& model_path);
  
  // crop image
  void Extract(const cv::Mat &image);

  // source image, detect box
  void Extract(const cv::Mat &image, const cv::Rect2f& rect);
  cv::Mat ComputeCropMatrix(const cv::Rect2f &rect);
  std::vector<cv::Point2f> getPoints() const;
  std::vector<cv::Point2f> getSourceImagePoints() const;

private:
  ModelInferRKNN m_infer_;
  std::vector<cv::Point2f> pts106_;
  std::vector<cv::Point2f> sourceimagepts;
};

#endif // FACE_LANDMARK_H
