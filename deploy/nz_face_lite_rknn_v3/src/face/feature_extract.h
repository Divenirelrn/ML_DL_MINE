//======================================================================================
//
//        Copyright (C) 2021 THL A29 Limited, a hangyi company. All rights reserved.
//
//        filename :feature_extract.h
//        description : feature extract alg on rv1109
//        created by yangjunpei and jackyu at  31/05/2021 18:19:28
//
//======================================================================================

#ifndef FEATURE_EXTRACT_H
#define FEATURE_EXTRACT_H

#include "face_preprocess.h"
#include "rknn_infer.h"
#include <opencv2/opencv.hpp>

using namespace std;
class FeatureExtract {
public:
  FeatureExtract();
  void Reset(ZipWrapper &wrapper);
  void Reset(std::string& model_path);

  // aligned image
  vector<float> Extract(cv::Mat &images);

  // aligned image
  vector<float> Extract(cv::Mat &images, float* feature);

  // source image
  vector<float> Extract(cv::Mat &images, std::vector<cv::Point2f>& landmarks, float* feature);

  void  Extract(cv::Mat &images, cv::Mat &feature);
  void  Alignment(cv::Mat &image, cv::Mat &transformed, cv::Rect rect, cv::Mat &ldk);
  float GetL2Norm();
  float GetSimilarScore(std::vector<float> feat1, std::vector<float> feat2);
  float GetSimilarScore(float* feat1, float* feat2);
  int   GetFeatureLence();
  cv::Mat GetAlignmentMatrix(std::vector<cv::Point2f> &pts);
  std::string tag;

private:
  ModelInferRKNN m_infer_;
  float L2_norm;
  int feature_lence = 512;
};

#endif // FEATURE_EXTRACT_H
