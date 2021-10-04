//======================================================================================
//
//        Copyright (C) 2021 THL A29 Limited, a hangyi company. All rights reserved.
//
//        filename :silent_live_ir.h
//        description : ir live alg on rv1109
//        created by yangjunpei and jackyu at  31/05/2021 18:19:28
//
//======================================================================================

#ifndef SILENT_LIVE_IR_H
#define SILENT_LIVE_IR_H

#include "rknn_infer.h"
#include "face_preprocess.h"
#include <opencv2/opencv.hpp>
#include "retinafacev2.h"
#include "face_quality.h"

class SilentLiveIR {
public:
  SilentLiveIR();
  void Reset(ZipWrapper &wrapper);
  void Reset(std::string& model_path);
  cv::Mat ComputeCropMatrix(const cv::Rect2f& rect);

  // for crop image
  float Check(const cv::Mat &crop_img);

  // for source images and detet box. 
  // trans, enpand, cutw_ratio, cuth_ratio: the displacement of ir image and rgb image, save imgage
  float Check(const cv::Mat &img, RetinaFaceV2& face_detection, const cv::Rect2f& rect, float trans, float expand, float cutw_ratio, float cuth_ratio, std::string outName);

  // trans, enpand, cutw_ratio, cuth_ratio: the displacement of ir image and rgb image 
  // float Check(const cv::Mat &img, RetinaFaceV2& face_detection, const cv::Rect2f& rect, float trans, float expand, float cutw_ratio, float cuth_ratio);
  
  int ir_Alignment_112(const cv::Mat &image, cv::Mat &transformed, cv::Mat &ldk);
  
  // algined, trans, enpand, cutw_ratio, cuth_ratio: the displacement of ir image and rgb image
  float Check(const cv::Mat &img, RetinaFaceV2& face_detection, FaceQuality& face_quality, const cv::Rect2f& rect, float trans, float expand, float cutw_ratio, float cuth_ratio);

  // use ir_img, detection rect, use use ComputeCropMatrix save
  float Check(const cv::Mat &img, const cv::Rect2f& rect, std::string outName);

  // use ir_img, detection rect, use use ComputeCropMatrix
  float Check(const cv::Mat &img, const cv::Rect2f& rect);

  // use ir_img, pose model landmarks algined  save crop image
  float Check(const cv::Mat &img, std::vector<cv::Point2f>& real_points, std::string outName);

  // use ir_img, pose model landmarks algined
  float Check(const cv::Mat &img, std::vector<cv::Point2f>& real_points);
private:
  ModelInferRKNN m_infer_;
};

#endif // SILENT_LIVE_IR_H
