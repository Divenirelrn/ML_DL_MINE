//======================================================================================
//
//        Copyright (C) 2021 THL A29 Limited, a hangyi company. All rights reserved.
//
//        filename :feature_extract.cpp
//        description : feature extract alg on rv1109
//        created by yangjunpei and jackyu at  31/05/2021 18:19:28
//
//======================================================================================

#include "feature_extract.h"
#include <math.h>

FeatureExtract::FeatureExtract() {
}

void FeatureExtract::Reset(ZipWrapper &wrapper) {
  std::string model_config = wrapper.ReadFile("models.yaml");
  Yaml::Node root;
  Yaml::Parse(root, model_config);
  ModelConfig config;
  config.FromYmalNode(root["feature"]);
  m_infer_.InitRunner(config, &wrapper);
}

void FeatureExtract::Reset(std::string& model_path) {
  ZipWrapper yaml_wrapper(model_path + "/libmodels.so");
  std::string model_config = yaml_wrapper.ReadFile("models.yaml");

  Yaml::Node root;
  Yaml::Parse(root, model_config);
  ModelConfig config;
  config.FromYmalNode(root["feature"]);

  std::string modelfile_name = model_path + "/" + config.filename; //"/r50_face.rknn";
  m_infer_.InitRunner(config, modelfile_name);
}

vector<float> FeatureExtract::Extract(cv::Mat &image) {
  cv::Mat image_bgr;
  cv::cvtColor(image, image_bgr, cv::COLOR_BGR2RGB);

  m_infer_.SetInputData(0, image_bgr);
  m_infer_.RunModel();
  int size = m_infer_.GetOutputTensorLen(0);
  std::cout << "feature size:" << size << std::endl;
  const float *data_ptr_out = m_infer_.GetOutputData(0);
  //std::cout << "here1" <<std::endl;
  vector<float> infer_result(data_ptr_out, data_ptr_out + size);
  //std::cout << "here2" << std::endl;
#if 0
  //for (auto f : infer_result) {
  //  std::cout << f << " ";
  //}
  //std::cout << std::endl;
#endif
  //std::cout << "here3" << std::endl;

  float l2 = 0;
  for (auto &one : infer_result) {
    l2 += one * one;
  }
  l2 = sqrt(l2);
  L2_norm = l2;
  std::cout << "L2 norm:" << l2 << std::endl;
  for (int i = 0; i < infer_result.size(); i++) {
    infer_result[i] /= l2;
  }
  return infer_result;
}

vector<float> FeatureExtract::Extract(cv::Mat &image, float* feature) {
  cv::Mat image_bgr;
  cv::cvtColor(image, image_bgr, cv::COLOR_BGR2RGB);

  m_infer_.SetInputData(0, image_bgr);
  m_infer_.RunModel();
  int size = m_infer_.GetOutputTensorLen(0);
  std::cout << "feature size:" << size << std::endl;
  const float *data_ptr_out = m_infer_.GetOutputData(0);
  //std::cout << "here1" <<std::endl;
  vector<float> infer_result(data_ptr_out, data_ptr_out + size);
  //std::cout << "here2" << std::endl;
#if 0
  //for (auto f : infer_result) {
  //  std::cout << f << " ";
  //}
  //std::cout << std::endl;
#endif
  //std::cout << "here3" << std::endl;

  float l2 = 0;
  for (auto &one : infer_result) {
    l2 += one * one;
  }
  l2 = sqrt(l2);
  L2_norm = l2;
  std::cout << "L2 norm:" << l2 << std::endl;
  for (int i = 0; i < infer_result.size(); i++) {
    infer_result[i] /= l2;
    feature[i]       = infer_result[i];
  }
  return infer_result;
}

vector<float> FeatureExtract::Extract(cv::Mat &image, std::vector<cv::Point2f>& landmarks, float* feature) {
  float lds[10] = {0,};
  for (int j=0; j<landmarks.size(); j++) {
    lds[2*j]     = landmarks[j].x;
    lds[2*j + 1] = landmarks[j].y;
  }
  cv::Mat detlds(5, 2, CV_32F);
  detlds.data = (unsigned char *)lds;
  cv::Mat algined;
  Alignment(image, algined, cv::Rect(2,3,10,10), detlds);

  //cv::imwrite("aligned_crop_putnet.jpg", algined);
  cv::Mat image_bgr;
  cv::cvtColor(algined, image_bgr, cv::COLOR_BGR2RGB);

  m_infer_.SetInputData(0, image_bgr);
  m_infer_.RunModel();
  int size = m_infer_.GetOutputTensorLen(0);
  std::cout << "feature size:" << size << std::endl;
  const float *data_ptr_out = m_infer_.GetOutputData(0);
  vector<float> infer_result(data_ptr_out, data_ptr_out + size);
  float l2 = 0;
  for (auto &one : infer_result) {
    l2 += one * one;
  }
  l2 = sqrt(l2);
  L2_norm = l2;
  std::cout << "L2 norm:" << l2 << std::endl;
  for (int i = 0; i < infer_result.size(); i++) {
    infer_result[i] /= l2;
    feature[i]       = infer_result[i];
  }
  return infer_result;
}

void FeatureExtract::Extract(cv::Mat &images, cv::Mat &feature) {
  cv::Mat image_bgr;
  cv::cvtColor(images, image_bgr, cv::COLOR_BGR2RGB);
  m_infer_.SetInputData(0, image_bgr);
  m_infer_.RunModel();
  int size = m_infer_.GetOutputTensorLen(0);
  const float *data_ptr_out = m_infer_.GetOutputData(0);
  vector<float> infer_result(data_ptr_out, data_ptr_out + size);
  cv::Mat to_mat = cv::Mat(infer_result);
  double dot = to_mat.dot(to_mat);
  feature = to_mat / sqrt(dot);
}

void FeatureExtract::Alignment(cv::Mat &image, cv::Mat &transformed, cv::Rect rect, cv::Mat &ldk) {
  float src_pts[] = {30.2946, 51.6963, 65.5318, 51.5014, 48.0252,
                     71.7366, 33.5493, 92.3655, 62.7299, 92.2041};
  int w = rect.width;
  int h = rect.height;
  cv::Size image_size(112, 112);
  cv::Size bounding_box_size(w, h);
  if (image_size.height) {
    for (int i = 0; i < 5; i++) {
      *(src_pts + 2 * i) += 8.0;
    }
  }
  cv::Mat src(5, 2, CV_32F);
  src.data = (uchar *)src_pts;
  cv::Mat M_temp = FacePreprocess::similarTransform(ldk, src);
  cv::Mat M = M_temp.rowRange(0, 2);
  cv::warpAffine(image, transformed, M, image_size, cv::INTER_CUBIC);
}

cv::Mat FeatureExtract::GetAlignmentMatrix(std::vector<cv::Point2f> &pts) {
  float src_pts[] = {30.2946, 51.6963, 65.5318, 51.5014, 48.0252,
                     71.7366, 33.5493, 92.3655, 62.7299, 92.2041};
  cv::Size image_size(112, 112);
  cv::Mat input_mat(5, 2, CV_32F, pts.data());
  // cv::Mat input_mat(5,2,CV_32F,pts.data());
  if (image_size.height) {
    for (int i = 0; i < 5; i++) {
      *(src_pts + 2 * i) += 8.0;
    }
  }
  cv::Mat src(5, 2, CV_32F);
  src.data = (uchar *)src_pts;
  cv::Mat M_temp = FacePreprocess::similarTransform(input_mat, src);
  cv::Mat M = M_temp.rowRange(0, 2);
  std::cout << "[AFFINE]:" << M.at<float>(0, 0) << " " << M.at<float>(0, 1)
            << " " << M.at<float>(0, 2) << " " << M.at<float>(1, 0) << " "
            << M.at<float>(1, 1) << " " << M.at<float>(1, 2) << std::endl;
  return M;
}

float FeatureExtract::GetSimilarScore(std::vector<float> feat1, std::vector<float> feat2) {
  assert(feat1.size() == feat2.size());
  float score = 0.0f;
  for (int i=0; i<feat1.size(); i++) {
    score += feat1[i] * feat2[i];
  }
  
  bool is_mapping = true;
  if (is_mapping) {
    return (score+1.0)/2.0f;
  } else {
    return score;
  }
}

float FeatureExtract::GetSimilarScore(float* feat1, float* feat2) {
  //assert(feat1.size() == feat2.size());
  float score = 0.0f;
  int feat_len = GetFeatureLence();
  for (int i=0; i<feat_len; i++) {
    score += feat1[i] * feat2[i];
  }
  
  bool is_mapping = true;
  if (is_mapping) {
    return (score+1.0)/2.0f;
  } else {
    return score;
  }
}

float FeatureExtract::GetL2Norm() { return L2_norm; }

int FeatureExtract::GetFeatureLence() { return feature_lence; }
