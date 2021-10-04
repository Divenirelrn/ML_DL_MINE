## 教程

#### 目前可以使用的

- [x] 人脸特征提取
- [ ] 人脸检测
- [ ] 人脸IR活体
- [x] 人脸对齐
- [ ] 人脸质量评估

#### 使用例子和解释 以人脸识别为例

```c++
#include "feature_extract.h" 
int main(int argc, char **argv) {
 if (argc < 4) {
   std::cout << argv[0] << " ";
   std::cout << "model_path pic_path  " << std::endl;
   return 0;
 }
  
  string model_path = argv[1];
  string model_path = argv[2];
  ZipWrapper zipWrapper(model_path); //模型打包在zip中，通过models.yaml定义不同的模型和模型预处理等。
  
  FeatureExtract featureExtract; // 构建人脸的 feature Extractor 
  featureExtract.Reset(zipWrapper); // 初始化
  

  cv::Mat image = cv::imread(path_list[idx]);
  std::vector<float> feature = featureExtract.Extract(image); // 得到的人脸特征

  return 0;
}
```

`