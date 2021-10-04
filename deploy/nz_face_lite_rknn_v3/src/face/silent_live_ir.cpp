//======================================================================================
//
//        Copyright (C) 2021 THL A29 Limited, a hangyi company. All rights reserved.
//
//        filename :silent_live_ir.cpp
//        description : ir live alg on rv1109
//        created by yangjunpei and jackyu at  31/05/2021 18:19:28
//
//======================================================================================

#include "silent_live_ir.h"
#include "retinafacev2.h"
#include "utils.h"
#include <utility>

SilentLiveIR::SilentLiveIR() {}

void SilentLiveIR::Reset(ZipWrapper &wrapper) {
  std::string model_config = wrapper.ReadFile("models.yaml");
  Yaml::Node root;
  Yaml::Parse(root, model_config);
  ModelConfig config;
  config.FromYmalNode(root["ir_live"]);
  m_infer_.InitRunner(config, &wrapper);
}

void SilentLiveIR::Reset(std::string& model_path) {
  ZipWrapper yaml_wrapper(model_path + "/libmodels.so");
  std::string model_config = yaml_wrapper.ReadFile("models.yaml");
  
  Yaml::Node root;
  Yaml::Parse(root, model_config);
  ModelConfig config;
  config.FromYmalNode(root["ir_live"]);

  std::string modelfile_name = model_path + "/" + config.filename; //"/liveness_ir.rknn";
  m_infer_.InitRunner(config, modelfile_name);
}

cv::Mat SilentLiveIR::ComputeCropMatrix(const cv::Rect2f &rect) {
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
  std::vector<cv::Point2f> dst_pts = {{0, 0}, {96, 0}, {96, 96}};
  cv::Mat m = cv::getAffineTransform(rect_pts, dst_pts);
  return m;
}

float SilentLiveIR::Check(const cv::Mat &crop_image) {
  assert(crop_image.rows == 96);
  assert(crop_image.cols == 96);
  m_infer_.SetInputData(0, crop_image);
  m_infer_.RunModel();
  const float *output = m_infer_.GetOutputData(0);
  std::vector<int> out_tensor_size = m_infer_.GetOutputTensorSize(0);
  int sum = 1;
  for (int i = 0; i < out_tensor_size.size(); i++) {
    sum *= out_tensor_size[i];
  }
  std::vector<float> res(output, output + sum);
  float _sum = 0;
  for (int i = 0; i < res.size(); i++)
    _sum += res[i];
  return _sum / 144.0;
}

float SilentLiveIR::Check(const cv::Mat &img, RetinaFaceV2& face_detection, const cv::Rect2f& rect, float trans, float expand, float cutw_ratio, float cuth_ratio, std::string outName) {
  
  // get the dispacement image
  // should done
  if (expand < 0.f || expand > 1.f || trans < 0.f || trans > 1.f) {
    printf("expand and trans are not valid, please check\n");
  }
  if (cutw_ratio < 0 || cuth_ratio < 0) {
    printf("cutw_ratio and cuth_ratio are not valid, please check\n");
  }
  int widthSource  = img.cols;
  int heightSource = img.rows;
  int x            = (int)rect.x;
  int y            = (int)rect.y;
  int width        = (int)rect.width;
  int height       = (int)rect.height;

  int dst_x;
  int dst_y;
  int dst_width;
  int dst_height;
  int delt = (int)(widthSource * trans);
  int ref  = (int)(widthSource * expand);

  bool isHyatleft = true;
  if (isHyatleft) {
    if ((x - ref + delt) <= 0) {
      dst_x = 0;
    } else {
      dst_x = x - ref + delt;
    }
    if ((y - ref) <= 0) {
      dst_y = 0;
    } else {
      dst_y = y - ref;
    }
    if (x + width + ref + delt >= widthSource) {
      dst_width = widthSource - dst_x;
    } else {
      dst_width = x + width + ref + delt - dst_x;
    }
    if (y + height + ref >= heightSource) {
      dst_height = heightSource - dst_y;
    } else {
      dst_height = y + height + ref - dst_y;
    }        
  } else {
    if ((x - ref - delt) <= 0) {
      dst_x = 0;
    } else {
      dst_x = x - ref - delt;
    }
    if ((y - ref) <= 0) {
      dst_y = 0;
    } else {
      dst_y = y - ref;
    }

    if (x + width + ref >= widthSource) {
      dst_width = widthSource - dst_x - delt;
    } else {
      dst_width = x + width + ref - dst_x - delt;
    }

    if (y + height + ref >= heightSource) {
      dst_height = heightSource - dst_y;
    } else {
      dst_height = y + height + ref - dst_y;
    }
  }

  cv::Rect roi;
  roi.x      = dst_x;
  roi.y      = dst_y;
  roi.width  = dst_width;
  roi.height = dst_height;
  
  // printf("silent live ir : dst x, y, w, h = %d, %d, %d, %d\n", roi.x, roi.y, roi.width, roi.height);

  // get real image for alvie detect and live
  cv::Mat crop_live          = img(roi).clone();
  //cv::imwrite("ir_roi_crop.jpg", crop_live);

  std::vector<BoxInfo> boxes = face_detection.Detect(crop_live, 0.6, 0.3);
  int face_num   = boxes.size();
  if (face_num <= 0) {
    // printf("not find face on ir image\n");
    return -1.0f;
  }
  // get maxarea face
  int max_area   = 0;
  int max_index  = 0;
  for (int i=0; i<face_num; i++) {
    int area = (boxes[i].x2-boxes[i].x1)*(boxes[i].y2-boxes[i].y1);
    if (area > max_area) {
        max_index = i;
        max_area  = area;
    }
  }
  const cv::Rect2f ir_rect(dst_x+boxes[max_index].x1, dst_y+boxes[max_index].y1, boxes[max_index].x2-boxes[max_index].x1, boxes[max_index].y2-boxes[max_index].y1);
  
  cv::Mat affine_alive = ComputeCropMatrix(ir_rect);
  cv::Mat crop_img_alive;
  cv::warpAffine(img, crop_img_alive, affine_alive, cv::Size(96,96));

  if (outName.length() > 5) {
    cv::imwrite(outName, crop_img_alive);  
  }
  //cv::imwrite("ir_crop_putnet.jpg", crop_img_alive);
  assert(crop_img_alive.rows == 96);
  assert(crop_img_alive.cols == 96);
  m_infer_.SetInputData(0, crop_img_alive);
  m_infer_.RunModel();
  const float *output = m_infer_.GetOutputData(0);
  std::vector<int> out_tensor_size = m_infer_.GetOutputTensorSize(0);
  int sum = 1;
  for (int i = 0; i < out_tensor_size.size(); i++) {
    sum *= out_tensor_size[i];
  }
  std::vector<float> res(output, output + sum);
  float _sum = 0;
  for (int i = 0; i < res.size(); i++) {
     _sum += res[i];
  }
  return _sum / 144.0;
}

int SilentLiveIR::ir_Alignment_112(const cv::Mat &image, cv::Mat &transformed, cv::Mat &ldk) {
  float src_pts[] = {30.2946, 51.6963, 65.5318, 51.5014, 48.0252,
		                       71.7366, 33.5493, 92.3655, 62.7299, 92.2041};
  cv::Size image_size(112, 112);
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
  return 0;
}

float SilentLiveIR::Check(const cv::Mat &img, RetinaFaceV2& face_detection, FaceQuality& face_quality, const cv::Rect2f& rect, float trans, float expand, float cutw_ratio, float cuth_ratio) {
  // get the dispacement image
  // should done
  if (expand < 0.f || expand > 1.f || trans < 0.f || trans > 1.f) {
    printf("expand and trans are not valid, please check\n");
  }
  if (cutw_ratio < 0 || cuth_ratio < 0) {
    printf("cutw_ratio and cuth_ratio are not valid, please check\n");
  }
  int widthSource  = img.cols;
  int heightSource = img.rows;
  int x            = (int)rect.x;
  int y            = (int)rect.y;
  int width        = (int)rect.width;
  int height       = (int)rect.height;

  int dst_x;
  int dst_y;
  int dst_width;
  int dst_height;
  int delt = (int)(widthSource * trans);
  int ref  = (int)(widthSource * expand);

  bool isHyatleft = true;
  if (isHyatleft) {
    if ((x - ref + delt) <= 0) {
      dst_x = 0;
    } else {
      dst_x = x - ref + delt;
    }
    if ((y - ref) <= 0) {
      dst_y = 0;
    } else {
      dst_y = y - ref;
    }
    if (x + width + ref + delt >= widthSource) {
      dst_width = widthSource - dst_x;
    } else {
      dst_width = x + width + ref + delt - dst_x;
    }
    if (y + height + ref >= heightSource) {
      dst_height = heightSource - dst_y;
    } else {
      dst_height = y + height + ref - dst_y;
    }        
  } else {
    if ((x - ref - delt) <= 0) {
      dst_x = 0;
    } else {
      dst_x = x - ref - delt;
    }
    if ((y - ref) <= 0) {
      dst_y = 0;
    } else {
      dst_y = y - ref;
    }

    if (x + width + ref >= widthSource) {
      dst_width = widthSource - dst_x - delt;
    } else {
      dst_width = x + width + ref - dst_x - delt;
    }

    if (y + height + ref >= heightSource) {
      dst_height = heightSource - dst_y;
    } else {
      dst_height = y + height + ref - dst_y;
    }
  }

  cv::Rect roi;
  roi.x      = dst_x;
  roi.y      = dst_y;
  roi.width  = dst_width;
  roi.height = dst_height;
  
  printf("silent live ir : dst x, y, w, h = %d, %d, %d, %d\n", roi.x, roi.y, roi.width, roi.height);

  // get real image for alvie detect and live
  cv::Mat crop_live          = img(roi).clone();
  //cv::imwrite("ir_roi_crop.jpg", crop_live);

  std::vector<BoxInfo> boxes = face_detection.Detect(crop_live, 0.6, 0.3);
  int face_num   = boxes.size();
  if (face_num <= 0) {
    printf("not find face on ir image\n");
    return -1.0f;
  }
  // get maxarea face
  int max_area   = 0;
  int max_index  = 0;
  for (int i=0; i<face_num; i++) {
    int area = (boxes[i].x2-boxes[i].x1)*(boxes[i].y2-boxes[i].y1);
    if (area > max_area) {
        max_index = i;
        max_area  = area;
    }
  }
  
  const cv::Rect2f ir_rect(dst_x+boxes[max_index].x1, dst_y+boxes[max_index].y1, boxes[max_index].x2-boxes[max_index].x1, boxes[max_index].y2-boxes[max_index].y1);
  face_quality.Extract(img, ir_rect);

  float yaw                            = face_quality.getYaw();
  float roll                           = face_quality.getRoll();
  float pitch                          = face_quality.getPitch();
  std::vector<float> quality_          = face_quality.getQuality();
  std::vector<cv::Point2f> points      = face_quality.getPoints();
  std::vector<cv::Point2f> real_points = face_quality.getSourceImagePoints();
  printf("ir api face angle and quality: %f, %f, %f, %f, %f, %f\n", yaw, roll, pitch, quality_[0], quality_[1], quality_[2]);

  cv::Mat img_show = img.clone();
  for (int i=0; i<real_points.size(); i++) {
    cv::circle(img_show, cv::Point(real_points[i].x, real_points[i].y), 1, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
  }
  cv::imwrite("ir_api_pose_landmarks.jpg", img_show);

  float lds[10] = {0,};
  for (int j=0; j<real_points.size(); j++) {
    lds[2*j]     = real_points[j].x;
    lds[2*j + 1] = real_points[j].y;
  }
  cv::Mat detlds(5, 2, CV_32F);
  detlds.data = (unsigned char *)lds;
  cv::Mat algined_ir;
  ir_Alignment_112(img, algined_ir, detlds);

  cv::resize(algined_ir ,algined_ir ,cv::Size(96,96));
  
  cv::Mat algined_ir_flip;
  cv::flip(algined_ir, algined_ir_flip, 1);
  //cv::Mat affine_alive = ComputeCropMatrix(ir_rect);
  //cv::Mat crop_img_alive;
  //cv::warpAffine(img, crop_img_alive, affine_alive, cv::Size(96,96));
  cv::imwrite("ir_api_algined.png", algined_ir);

  assert(algined_ir.rows == 96);
  assert(algined_ir.cols == 96);
  m_infer_.SetInputData(0, algined_ir);
  m_infer_.RunModel();
  const float *output = m_infer_.GetOutputData(0);
  std::vector<int> out_tensor_size = m_infer_.GetOutputTensorSize(0);
  int sum = 1;
  for (int i = 0; i < out_tensor_size.size(); i++) {
    sum *= out_tensor_size[i];
  }
  std::vector<float> res(output, output + sum);
  float _sum = 0;
  for (int i = 0; i < res.size(); i++) {
     _sum += res[i];
  }
  
  assert(algined_ir_flip.rows == 96);
  assert(algined_ir_flip.cols == 96);
  m_infer_.SetInputData(0, algined_ir_flip);
  m_infer_.RunModel();
  const float *output_flip = m_infer_.GetOutputData(0);
  std::vector<int> out_tensor_size_flip = m_infer_.GetOutputTensorSize(0);
  int sum_flip = 1;
  for (int i = 0; i < out_tensor_size_flip.size(); i++) {
	  sum_flip *= out_tensor_size_flip[i];
  }
  std::vector<float> res_flip(output_flip, output_flip + sum_flip);
  float _sum_flip = 0;
  for (int i = 0; i < res_flip.size(); i++) {
	_sum_flip += res_flip[i];
  }
  printf("ir live:%f, flip_score:%f, av:%f\n", _sum/144.0, _sum_flip/144.0, (_sum+_sum_flip)/288.0);
  return (_sum + _sum_flip) / 288.0;
}

// ir input, ir detection rect, use ComputeCropMatrix  save
float SilentLiveIR::Check(const cv::Mat &img, const cv::Rect2f& rect, std::string outName) {

  cv::Mat affine_alive = ComputeCropMatrix(rect);
  cv::Mat crop_img_alive;
  cv::warpAffine(img, crop_img_alive, affine_alive, cv::Size(96,96));
  // cv::imwrite("ir_ComputeCropMatrix.png", crop_img_alive);
  // cv::imwrite("ir_real_points_algined.png", algined_ir);
  // std::string outName = "../nir_rv_det_crop0/" + filename;
  if (outName.length() > 5){
    cv::imwrite(outName, crop_img_alive);
  }
  assert(crop_img_alive.rows == 96);
  assert(crop_img_alive.cols == 96);
  m_infer_.SetInputData(0, crop_img_alive);
  m_infer_.RunModel();
  const float *output = m_infer_.GetOutputData(0);
  std::vector<int> out_tensor_size = m_infer_.GetOutputTensorSize(0);
  int sum = 1;
  for (int i = 0; i < out_tensor_size.size(); i++) {
    sum *= out_tensor_size[i];
  }
  std::vector<float> res(output, output + sum);
  float _sum = 0;
  for (int i = 0; i < res.size(); i++) {
     _sum += res[i];
  }
  return _sum / 144.0;
}

// ir input, ir detection rect, use ComputeCropMatrix 
float SilentLiveIR::Check(const cv::Mat &img, const cv::Rect2f& rect) {

  cv::Mat affine_alive = ComputeCropMatrix(rect);
  cv::Mat crop_img_alive;
  cv::warpAffine(img, crop_img_alive, affine_alive, cv::Size(96,96));
  cv::imwrite("ir_ComputeCropMatrix.png", crop_img_alive);

  assert(crop_img_alive.rows == 96);
  assert(crop_img_alive.cols == 96);
  m_infer_.SetInputData(0, crop_img_alive);
  m_infer_.RunModel();
  const float *output = m_infer_.GetOutputData(0);
  std::vector<int> out_tensor_size = m_infer_.GetOutputTensorSize(0);
  int sum = 1;
  for (int i = 0; i < out_tensor_size.size(); i++) {
    sum *= out_tensor_size[i];
  }
  std::vector<float> res(output, output + sum);
  float _sum = 0;
  for (int i = 0; i < res.size(); i++) {
     _sum += res[i];
  }
  return _sum / 144.0;
}
// ir input, use detection rect and pose landmarks aligned save crop image , std::string &filename
float SilentLiveIR::Check(const cv::Mat &img, std::vector<cv::Point2f>& real_points, std::string outName) {
  
  float lds[10] = {0,};
  for (int j=0; j<real_points.size(); j++) {
    lds[2*j]     = real_points[j].x;
    lds[2*j + 1] = real_points[j].y;
  }
  cv::Mat detlds(5, 2, CV_32F);
  detlds.data = (unsigned char *)lds;
  cv::Mat algined_ir;
  ir_Alignment_112(img, algined_ir, detlds);

  cv::resize(algined_ir ,algined_ir ,cv::Size(96,96));
  
  cv::Mat algined_ir_flip;
  cv::flip(algined_ir, algined_ir_flip, 1);

  // cv::imwrite("ir_real_points_algined.png", algined_ir);
  // std::string outName = "../nir_rv_det_crop/" + filename;
  if (outName.length() > 5) {
    cv::imwrite(outName, algined_ir);
  }

  assert(algined_ir.rows == 96);
  assert(algined_ir.cols == 96);
  m_infer_.SetInputData(0, algined_ir);
  m_infer_.RunModel();
  const float *output = m_infer_.GetOutputData(0);
  std::vector<int> out_tensor_size = m_infer_.GetOutputTensorSize(0);
  int sum = 1;
    for (int i = 0; i < out_tensor_size.size(); i++) {
	        sum *= out_tensor_size[i];
    }

    std::vector<float> res(output, output + sum);
    float _sum = 0;
    for (int i = 0; i < res.size(); i++) {
		       _sum += res[i];
    }

  assert(algined_ir_flip.rows == 96);
  assert(algined_ir_flip.cols == 96);
  m_infer_.SetInputData(0, algined_ir_flip);
  m_infer_.RunModel();
  const float *output_flip = m_infer_.GetOutputData(0);
  //std::vector<int> out_tensor_size = m_infer_.GetOutputTensorSize(0);
  
  std::vector<float> res_flip(output_flip, output_flip + sum);
  float _sum_flip = 0;
  for (int i = 0; i < res_flip.size(); i++) {
	  _sum_flip += res_flip[i];
  }
  float av_score = (_sum+_sum_flip)/288.0;
  // printf("normal score:%f, flip_score:%f, av:%f\n", _sum/144.0, _sum_flip/144.0, av_score);
  return av_score;

}

// ir input, use detection rect and pose landmarks aligned
float SilentLiveIR::Check(const cv::Mat &img, std::vector<cv::Point2f>& real_points) {
  
  float lds[10] = {0,};
  for (int j=0; j<real_points.size(); j++) {
    lds[2*j]     = real_points[j].x;
    lds[2*j + 1] = real_points[j].y;
  }
  cv::Mat detlds(5, 2, CV_32F);
  detlds.data = (unsigned char *)lds;
  cv::Mat algined_ir;
  ir_Alignment_112(img, algined_ir, detlds);

  cv::resize(algined_ir ,algined_ir ,cv::Size(96,96));

  cv::imwrite("ir_real_points_algined.png", algined_ir);

  assert(algined_ir.rows == 96);
  assert(algined_ir.cols == 96);
  m_infer_.SetInputData(0, algined_ir);
  m_infer_.RunModel();
  const float *output = m_infer_.GetOutputData(0);
  std::vector<int> out_tensor_size = m_infer_.GetOutputTensorSize(0);
  int sum = 1;
  for (int i = 0; i < out_tensor_size.size(); i++) {
    sum *= out_tensor_size[i];
  }
  std::vector<float> res(output, output + sum);
  float _sum = 0;
  for (int i = 0; i < res.size(); i++) {
     _sum += res[i];
  }
  return _sum / 144.0;
}




