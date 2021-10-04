#include "mobilenet.h"

//v0.1.0
//Initial version

std::string HyObjectVersion() {
    return "v0.1.0";
}

//Model reset
void MobileNet::Reset(ZipWrapper &wrapper) {
    input_size_ = 224;
    std::string model_config = wrapper.ReadFile("models.yaml");
    Yaml::Node root;
    Yaml::Parse(root, model_config);
    ModelConfig config;
    config.FromYmalNode(root["fatigue_recog"]);
    Net.InitRunner(config, &wrapper);
}


void MobileNet::Reset(std::string& model_path) {
  ZipWrapper yaml_wrapper(model_path + "/libmodels.so");
  input_size_ = 224;
  std::string model_config = yaml_wrapper.ReadFile("models.yaml");
  Yaml::Node root;
  Yaml::Parse(root, model_config);
  ModelConfig config;
  config.FromYmalNode(root["fatigue_recog"]);
  std::string modelfile_name = model_path + "/" + config.filename; //"/r18noscale.hw.rknn";
  Net.InitRunner(config, modelfile_name);
  std::cout << "Reset model success!" << std::endl;
}


//Decompose
MobileNet::~MobileNet() {}


//Preprocess
int MobileNet::PreProcess(const cv::Mat &image, cv::Mat &blob, bool is_quantized) {

  int input_w = input_size_;
  int input_h = input_size_;
  cv::Mat input_image;
  //Resize
  cv::resize(image, input_image, cv::Size(input_w, input_h));
  //BGR -> RGB
  if (!is_quantized) {
	  std::cout << "error only support quantized model!\n";
  } else {
      cv::cvtColor(input_image, blob, cv::COLOR_BGR2RGB);
  }

  return 0;
}

//Postprocess
int MobileNet::PostProcess(const std::vector<ClassInfo> &results, ClassInfo &max_res){
  float max_score = 0.0;
  int max_idx = -1;
  //Get max
  for(auto it = results.begin(); it != results.end(); it++){
    if((*it).cls_score > max_score){
      max_idx = (*it).cls_idx;
      max_score = (*it).cls_score;
    }
  } 
  max_res.cls_idx = max_idx;
  max_res.cls_score = max_score;

  return 0;
}

//Exp
inline float fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

//Softmax
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

//Infer
ClassInfo MobileNet::Infer(const cv::Mat &image) {
  cv::Mat input;
  //Proprocess
  int ret = PreProcess(image, input, true);

  //Model initial & inference
  Net.SetInputData(0, input);
  Net.RunModel();

  //Get output
  const std::vector<int> tsize_cls_pred = Net.GetOutputTensorSize(0);
  int cls_num = tsize_cls_pred.size();

  float src[cls_num];
  float dst[cls_num];
  for(int i = 0; i < cls_num; i++){
    const float *p = Net.GetOutputData(i);
    src[i] = *p;
  }
  
  //Output score through softmax
  ret = activation_function_softmax<float>(src, dst, cls_num);

  std::vector<ClassInfo> results;
  ClassInfo res;
  for(int i = 0; i < cls_num; i++){
    res.cls_idx = i;
    res.cls_score = dst[i];
    results.push_back(res);
  }
  //for(auto it = results.begin(); it != results.end(); it++){
  //    std::cout << (*it).cls_idx << ", " << (*it).cls_score << "\n";
  //}

  //Postprocess
  ClassInfo max_res;
  ret = PostProcess(results, max_res);

  return max_res;
}

