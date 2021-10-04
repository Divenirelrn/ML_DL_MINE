
#ifndef RETINAFACEV2_H
#define RETINAFACEV2_H

#include "rknn_infer.h"
#include <opencv2/opencv.hpp>

struct HeadInfo {
  std::string cls_layer;
  std::string dis_layer;
  int stride;
};

struct object_rect {
  int x;
  int y;
  int width;
  int height;
};

typedef struct BoxInfo {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int label;
  float cls_score;
  int cls_idx;
} BoxInfo;

class RetinaFaceV2 {
public:
  RetinaFaceV2() {} ;
  void Reset(ZipWrapper &wrapper);
  void Reset(std::string& model_path);

  ~RetinaFaceV2();

  //  static RetinaFaceV2 *detector;
  //  cv::dnn::Net Net;
  ModelInferRKNN Net;
  std::vector<HeadInfo> heads_info{
      // cls_pred|dis_pred|stride
      {"574", "577", 8},  {"590", "593", 16},  {"606", "609", 32},
      {"622", "625", 64}, {"638", "641", 128},
  };

  std::vector<BoxInfo> Detect(const cv::Mat &image, float score_threshold,
                              float nms_threshold);

private:
  float PreProcess(const cv::Mat &image, cv::Mat &in, bool is_quantized);

  void DecodeInfer(cv::Mat &cls_pred, cv::Mat &dis_pred, int stride,
                   float threshold, std::vector<BoxInfo> &results);

    void DecodeInfer_F(const std::vector<float> &cls_pred , const std::vector<float> &dis_pred ,  int stride,
                     float threshold, std::vector<BoxInfo> &results);

  BoxInfo DisPred2Bbox(const float *dfl_det, int label, float score, int x,
                       int y, int stride);
    //BoxInfo DisPred2Bbox(const float *&dfl_det, int label, float score, int x,
    //                     int y, int stride);

  static void nms(std::vector<BoxInfo> &result, float nms_threshold);

private:
  int input_size_;
  int reg_max = 0;
  void NHWC2HCHW(const cv::Mat &nwhc, cv::Mat &ncwh);
  // GestCls gestCls;
};

std::string HyFaceVersion();

#endif
