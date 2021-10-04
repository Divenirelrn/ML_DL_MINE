#ifndef FATIGUE_LIB_MOBILENET_H
#define FATIGUE_LIB_MOBILENET_H

#include "rknn_infer.h"
#include <opencv2/opencv.hpp>

//ClassInfo define
typedef struct ClassInfo {
    int cls_idx;
    float cls_score;
} ClassInfo;

class MobileNet {
public:
    MobileNet() {};
    void Reset(ZipWrapper &wrapper); //Model reset
    void Reset(std::string& model_path); 
    ClassInfo Infer(const cv::Mat &image); //Model inference

    ~MobileNet();

    ModelInferRKNN Net;

private:
    int PreProcess(const cv::Mat &image, cv::Mat &blob, bool is_quantized); //Preprocess
    int PostProcess(const std::vector<ClassInfo> &results, ClassInfo &max_res); //Postprocess

    int input_size_; //Input image size
};


#endif
