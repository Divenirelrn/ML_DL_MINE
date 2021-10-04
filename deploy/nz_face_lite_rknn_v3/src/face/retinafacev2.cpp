#include "retinafacev2.h"
#include "npy.hpp"

// v2.0.1 
// 1: the quality_ vector not clear bug by jpyang at: 2021-06-18-15:26
// 2: close the auth for test
// 3: version:v2.0.2 ---- creat rknn_infer.cpp file at:2021-07-05:16:47
// 4: version v2.0.3 ---- creat encryption          at:2021-07-06:15:12
// 5: version v2.0.4 ---- open auth                 at:2021-07-06:15:46
// 6: version v2.0.5 ---- create .so and .a lib for yufan   at:2021-07-06:09:51

// 7: version v3.0.0 ---- add nazhi rgb live alg by jpyang at:2021-08-12:17:21
// 8: version v3.0.1 ---- add nazhi rgb flip op and silent_ir pose model by jpyang at:2021-08-15:14:36

std::string HyFaceVersion() {
   return "v3.0.1";
}

inline float fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + fast_exp(-x)); }

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length) {
  const _Tp alpha = *std::max_element(src, src + length);
  _Tp denominator{0};

  for (int i = 0; i < length; ++i) {
    dst[i] = fast_exp(src[i] - alpha);
    denominator += dst[i];
  }

  for (int i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }
  return 0;
}

void RetinaFaceV2::Reset(ZipWrapper &wrapper)  {
  input_size_ = 256;
  std::string model_config = wrapper.ReadFile("models.yaml");
  Yaml::Node root;
  Yaml::Parse(root, model_config);
  ModelConfig config;
  config.FromYmalNode(root["face_detect"]);
  Net.InitRunner(config, &wrapper);
}

void RetinaFaceV2::Reset(std::string& model_path) {
  ZipWrapper yaml_wrapper(model_path + "/libmodels.so");
  input_size_ = 256;
  std::string model_config = yaml_wrapper.ReadFile("models.yaml");
  Yaml::Node root;
  Yaml::Parse(root, model_config);
  ModelConfig config;
  config.FromYmalNode(root["face_detect"]);
  std::string modelfile_name = model_path + "/" + config.filename; //"/r18noscale.hw.rknn";
  Net.InitRunner(config, modelfile_name);
}

RetinaFaceV2::~RetinaFaceV2() {}

float RetinaFaceV2::PreProcess(const cv::Mat &image, cv::Mat &blob,
                               bool is_quantized) {

  int input_w = input_size_;
  int input_h = input_size_;
  int w, h, x, y;
  float r_w = float(input_w) / image.cols;
  float r_h = float(input_h) / image.rows;
  if (r_h > r_w) {
    w = input_w;
    h = r_w * static_cast<float>(image.rows);
    x = 0;
    y = (input_h - h) / 2;
  } else {
    w = r_h * image.cols;
    h = input_h;
    x = (input_w - w) / 2;
    y = 0;
  }
  cv::Mat re(h, w, CV_8UC3);
  float det_scale = float(re.cols) / image.cols;
  cv::resize(image, re, re.size(), 0, 0, cv::INTER_CUBIC);
  cv::Mat det_img(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
  re.copyTo(det_img(cv::Rect(0, 0, re.cols, re.rows)));
  if (!is_quantized) {
	  printf("error only support quantized model.\n");
  } else {
      cv::cvtColor(det_img, blob , cv::COLOR_BGR2RGB);
//    blob = det_img;
  }

  //cv::imwrite("blob.jpg" ,blob);
  return det_scale;
}


void nhwc2nchw(const std::vector<float> &data , std::vector<int> input_sizes, std::vector<float> &out ){
    out.resize(data.size());
    assert(input_sizes.size() ==4);
    assert(input_sizes[0] == 1);
    int batch = input_sizes[0];
    int h = input_sizes[1];
    int w = input_sizes[2];
    int c = input_sizes[3];
    for(int c  = 0 ; c < c ; c++)
        for (int h = 0 ; h < h ; h++)
            for(int w = 0 ; w < w ; w++)
                out[c*w*h +  h*w + w] = data[h*w*c + w*c + c];
}

std::vector<int> nhwc2nchw_size(std::vector<int> input_sizes) {
    assert(input_sizes.size() ==4);
    assert(input_sizes[0] == 1);
    int batch = input_sizes[0];
    int h = input_sizes[1];
    int w = input_sizes[2];
    int c = input_sizes[3];
    std::vector<int> out_size;
    out_size.push_back(batch);
    out_size.push_back(c);
    out_size.push_back(h);
    out_size.push_back(w);
    return out_size;
}

//std::vector<int> transpose(std::vector<int> intput_size , )

int count_size(const std::vector<int> &input_sizes){
    int sum = 1;
    for(auto one:input_sizes){
        sum*=one;
    }
    return sum;
}


std::vector<BoxInfo> RetinaFaceV2::Detect(const cv::Mat &image,
                                          float score_threshold,
                                          float nms_threshold) {
  cv::Mat input;
  float det_scale = PreProcess(image, input, true);
  Net.SetInputData(0, input);
  std::vector<std::string> nodes;
  for (const auto &head_info : this->heads_info) {
    nodes.push_back(head_info.cls_layer.c_str());
    nodes.push_back(head_info.dis_layer.c_str());
  }
  int fmc = nodes.size() / 2;
  std::vector<BoxInfo> results;
  std::vector<cv::Mat> outputs;
  Net.RunModel();

for (int i = 0; i < fmc; i++) {
    const std::vector<int> tsize_cls_pred = Net.GetOutputTensorSize(i * 2);
    //printf("fmc cls:%d size:%d    " , i  , tsize_cls_pred.size());
    //for(auto one:tsize_cls_pred)
    //    printf(" %d " , one);
    //printf("\n");
    int stride = heads_info[i].stride;
    int blob_count_cls_pred = count_size(tsize_cls_pred);
    std::vector<float>  data_cls_pred(Net.GetOutputData(i * 2), Net.GetOutputData(i * 2)+blob_count_cls_pred);
    const std::vector<int> tsize_dis_pred = Net.GetOutputTensorSize(i * 2 + 1);
    int blob_count_dis_pred = count_size(tsize_dis_pred);
    std::vector<float>  data_dis_pred(Net.GetOutputData(i * 2 + 1), Net.GetOutputData(i * 2 + 1 )+blob_count_dis_pred);
    DecodeInfer_F(data_cls_pred, data_dis_pred, stride, score_threshold, results);
  }
  for (int i = 0; i < results.size(); i++) {
    auto &box = results[i];
    box.x1 /= det_scale;
    box.x2 /= det_scale;
    box.y1 /= det_scale;
    box.y2 /= det_scale;
  }
  nms(results, nms_threshold);
  return results;
}

inline float *rows(const cv::Mat &cls_pred, const int y) {
  // N H W
  int w = cls_pred.size[2];
  return (float *)((unsigned char *)cls_pred.data + (size_t)w * y * 4);
}


void RetinaFaceV2::DecodeInfer_F(const std::vector<float> &cls_pred, const std::vector<float> &dis_pred, int stride,
                               float threshold, std::vector<BoxInfo> &results) {
  int feature_h = input_size_ / stride;
  int feature_w = input_size_ / stride;
  int feature_size = feature_h * feature_w;
  for (int idx = 0; idx < feature_size ;  idx++) {
    float score = cls_pred[idx];
    int cur_label = 0;
        int row = idx / feature_w;
        int col = idx % feature_w;

      if (score >= threshold) {
         const float *bbox = dis_pred.data() + idx*4;
          results.push_back(DisPred2Bbox(bbox, cur_label, score, col, row, stride));
    }
  }
}

//void RetinaFaceV2::DecodeInfer(cv::Mat &cls_pred, cv::Mat &dis_pred, int stride,
//                               float threshold, std::vector<BoxInfo> &results) {
//    int feature_h = input_size_ / stride;
//    int feature_w = input_size_ / stride;
//    // std::cout<<"A:"<<cls_pred.size<<","<<dis_pred.size<<","<<feature_h<<","<<feature_w<<std::endl;
//    // cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
//    float *data_out = (float *)cls_pred.data;
//    for (int idx = 0; idx < feature_h * feature_w; idx++) {
//        const float *scores = rows(cls_pred, idx);
//        int row = idx / feature_w;
//        int col = idx % feature_w;
//        float score = scores[0];
//        // std::cout<<"S:"<<score<<","<<cls_pred.at<float>(idx)<<std::endl;
//        int cur_label = 0;
//        if (score >= threshold) {
//            const float *bbox_pred = rows(dis_pred, idx);
//            results.push_back(
//                    DisPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
//        }
//    }
//}


BoxInfo RetinaFaceV2::DisPred2Bbox(const float *dfl_det, int label,
                                   float score, int x, int y, int stride) {
  // float ct_x = (x + 0.5) * stride;
  // float ct_y = (y + 0.5) * stride;
  float ct_x = (x + 0.0) * stride;
  float ct_y = (y + 0.0) * stride;
  std::vector<float> dis_pred;
  dis_pred.resize(4);
  for (int i = 0; i < 4; i++) {
    dis_pred[i] = (*(dfl_det + i)) * stride;
    // float dis = 0;
    // float *dis_after_sm = new float[this->reg_max + 1];
    // activation_function_softmax(dfl_det + i * (this->reg_max + 1),
    // dis_after_sm,
    //                            this->reg_max + 1);
    // for (int j = 0; j < this->reg_max + 1; j++) {
    //  dis += j * dis_after_sm[j];
    //}
    // dis *= stride;
    // dis_pred[i] = dis;
    // delete[] dis_after_sm;
  }
  // float xmin = (std::max)(ct_x - dis_pred[0], .0f);
  // float ymin = (std::max)(ct_y - dis_pred[1], .0f);
  // float xmax = (std::min)(ct_x + dis_pred[2], (float)this->input_size);
  // float ymax = (std::min)(ct_y + dis_pred[3], (float)this->input_size);
  float xmin = ct_x - dis_pred[0];
  float ymin = ct_y - dis_pred[1];
  float xmax = ct_x + dis_pred[2];
  float ymax = ct_y + dis_pred[3];
  return BoxInfo{xmin, ymin, xmax, ymax, score, label};
}

void RetinaFaceV2::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
  std::sort(input_boxes.begin(), input_boxes.end(),
            [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
  std::vector<float> vArea(input_boxes.size());
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) *
               (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    for (int j = i + 1; j < int(input_boxes.size());) {
      float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
      float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
      float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
      float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
      float w = (std::max)(float(0), xx2 - xx1 + 1);
      float h = (std::max)(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= NMS_THRESH) {
        input_boxes.erase(input_boxes.begin() + j);
        vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }
    }
  }
}

int resize_uniform(const cv::Mat &src, cv::Mat &dst, cv::Size dst_size,
                   object_rect &effect_area) {
  int w = src.cols;
  int h = src.rows;
  int dst_w = dst_size.width;
  int dst_h = dst_size.height;
  dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

  float ratio_src = w * 1.0 / h;
  float ratio_dst = dst_w * 1.0 / dst_h;

  int tmp_w = 0;
  int tmp_h = 0;
  if (ratio_src > ratio_dst) {
    tmp_w = dst_w;
    tmp_h = floor((dst_w * 1.0 / w) * h);
  } else if (ratio_src < ratio_dst) {
    tmp_h = dst_h;
    tmp_w = floor((dst_h * 1.0 / h) * w);
  } else {
    cv::resize(src, dst, dst_size);
    effect_area.x = 0;
    effect_area.y = 0;
    effect_area.width = dst_w;
    effect_area.height = dst_h;
    return 0;
  }
  cv::Mat tmp;
  cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));
  if (tmp_w != dst_w) {
    int index_w = floor((dst_w - tmp_w) / 2.0);
    for (int i = 0; i < dst_h; i++) {
      memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3,
             tmp_w * 3);
    }
    effect_area.x = index_w;
    effect_area.y = 0;
    effect_area.width = tmp_w;
    effect_area.height = tmp_h;
  } else if (tmp_h != dst_h) {
    int index_h = floor((dst_h - tmp_h) / 2.0);
    memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
    effect_area.x = 0;
    effect_area.y = index_h;
    effect_area.width = tmp_w;
    effect_area.height = tmp_h;
  } else {
    printf("error\n");
  }
  return 0;
}
