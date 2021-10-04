#ifndef LMKTRACKING_LIB_MODEL_INFER_H
#define LMKTRACKING_LIB_MODEL_INFER_H

#include "Yaml.hpp"
#include "opencv2/opencv.hpp"
#include "zip_wrapper.h"
#include <vector>

enum ModelType {
  MODEL_UNKNOWN = 0,
  MODEL_MNN = 1,
  MODEL_OPENCV = 2,
  MODEL_NCNN = 3,
  MODEL_RKNN = 4,
};

enum Status { SUCCESS = 0, ERROR_SHAPE_MATCH = 1, ERROR_DATA_ORDER = 2 };

struct ModelConfig {
  ModelType type = MODEL_RKNN;
  int threads = 1;
  std::string version;
  std::string name;
  std::string filename;
  std::vector<float> norm_vals;
  std::vector<float> mean_vals;
  std::vector<std::string> input_nodes;
  std::vector<std::string> output_nodes;
  std::string input_data_order;
  std::string input_data_type;
  std::string infer_engine;
  std::string infer_backend;
  std::string infer_device;
  std::string model_type;
  Yaml::Node *node;
  void FromYmalNode(Yaml::Node &node_value) {
    name = node_value["name"].As<std::string>();
    filename = node_value["file_name"].As<std::string>();
    version = node_value["version"].As<std::string>();
    threads = node_value["threads"].As<int>();
    model_type = node_value["model_type"].As<std::string>();
    infer_engine = node_value["infer_engine"].As<std::string>();
    infer_device = node_value["infer_deivce"].As<std::string>();
    infer_backend = node_value["infer_backend"].As<std::string>();
    input_data_order = node_value["input_data_order"].As<std::string>();
    input_data_type = node_value["input_data_type"].As<std::string>();
    input_nodes.resize(node_value["input_nodes"].Size());
    for (int i = 0; i < node_value["input_nodes"].Size(); i++) {
      input_nodes[i] = node_value["input_nodes"][i].As<std::string>();
    }
    output_nodes.resize(node_value["output_nodes"].Size());
    for (int i = 0; i < node_value["output_nodes"].Size(); i++) {
      output_nodes[i] = node_value["output_nodes"][i].As<std::string>();
    }
    float size = node_value["a"][0].As<float>();
    norm_vals.resize(node_value["norm_vals"].Size());
    for (int i = 0; i < node_value["norm_vals"].Size(); i++) {
      norm_vals[i] = node_value["norm_vals"][i].As<float>();
    }
    mean_vals.resize(node_value["mean_vals"].Size());
    for (int i = 0; i < node_value["mean_vals"].Size(); i++) {
      mean_vals[i] = node_value["mean_vals"][i].As<float>();
    }
    node = &node_value;
  }

  std::string ToInfo() const {
    std::stringstream ss;
    return ss.str();
  }
};

class ModelRunner {
public:
  virtual ~ModelRunner() {}
  virtual int InitRunner(const ModelConfig &config, ZipWrapper *wrapper) = 0;
  virtual ModelType GetModelType() const = 0;
  virtual std::vector<int>
  GetInputTensorSize(const std::string &index_name) const = 0;
  virtual std::vector<int>
  GetOutputTensorSize(const std::string &index_name) const = 0;
  virtual Status SetInputData(int index, const void *data) = 0;
  //  virtual Status SetInputData(const std::string &index_name, const void
  //  *data) = 0;
  virtual Status SetInputData(const std::string &index_name,
                              const cv::Mat &image) = 0;
  virtual void RunModel() = 0;
  virtual const void *GetOutputData(int index) = 0;
  virtual const float *GetOutputData(const std::string &index_name) = 0;
  virtual void ResizeInputTensor(const std::string &index_name,
                                 const std::vector<int> &shape) = 0;
};

// using ModelPtr = std::shared_ptr<Model>;

#endif // LMKTRACKING_LIB_MODEL_INFER_H
