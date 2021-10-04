
#include "model_infer.h"
#include "opencv2/opencv.hpp"
#include "rknn_api.h"
#include "zip_wrapper.h"
#include <pthread.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <string.h>
#include "rknn_infer.h"

#define AUTH_CHECK 1

#if AUTH_CHECK
// used for authentication
static const char *socket_path = "server.socket";  
static int  init_call = 0;
static int  check_done = 0;

// auth check handle
static void *  auth_check(void * arg)
{ 
    struct sockaddr_un address;  
    address.sun_family = AF_UNIX;  
    strcpy(address.sun_path, socket_path);  
    
    struct timeval tv;    
    
    while(1)
    {         
        /* create a socket */  
        int sockfd = socket(AF_UNIX, SOCK_STREAM, 0);  
        /* connect to the server */
        int result = connect(sockfd, (struct sockaddr *)&address, sizeof(address));  
        if(result == -1)   
        {  
            perror("connect socket file failed");  
            exit(-1);  
        }  
        char secret[256] = "";
        int n = read(sockfd, secret, 256);  
        //printf("get info from server: %s\n", secret);  
        
        /* close the socket */       
        close(sockfd);
        
        char    begin_time[16] = "";
        char    end_time[16] = "";
        
        strncpy(begin_time, secret+13, 10);
        strncpy(end_time,   secret+23, 10);
        
        long lbtime = atol(begin_time);
        long lttime = atol(end_time);
        
        gettimeofday(&tv, NULL);
        
        //printf("current second: %ld, begin_time:%ld, end_time:%ld\n", tv.tv_sec, lbtime, lttime);
        
        if(tv.tv_sec >= lbtime && tv.tv_sec <= lttime )
            printf("auth successful, go ahead  \n");
        else
        {
            printf("auth expired, pls contact admin \n");
            exit(-1);
        }
        check_done = 1;
        sleep(5);
    }       
}
#endif

unsigned char key[256] = {
    0xAB, 0x7C, 0x3A, 0x24, 0xE9, 0x3E, 0xC4, 0x55, 0x29, 0x46, 0x96, 0x63,
    0xB2, 0xCF, 0xCC, 0x88, 0x28, 0xFD, 0xCC, 0x27, 0xB1, 0x71, 0x84, 0x02,
    0x57, 0x1C, 0xF3, 0xAD, 0xAE, 0x50, 0xEB, 0x43, 0xB6, 0x1D, 0x15, 0x57,
    0x02, 0xDA, 0x35, 0x2D, 0x1A, 0x02, 0xAF, 0x2A, 0x1F, 0x42, 0x4B, 0x8A,
    0xAD, 0xCF, 0x35, 0x51, 0x05, 0xFD, 0x48, 0x84, 0x6C, 0x0E, 0x8B, 0x98,
    0xCE, 0x47, 0xFC, 0x2B, 0xA7, 0x4A, 0x8D, 0xF2, 0x24, 0x9F, 0x6D, 0xF3,
    0x85, 0x94, 0x88, 0xF4, 0xC5, 0x47, 0x51, 0x33, 0x7D, 0x01, 0xBF, 0x57,
    0x06, 0xC5, 0x95, 0xA7, 0xDD, 0x29, 0xE7, 0x7A, 0x4C, 0x65, 0xDA, 0xEE,
    0x47, 0xAA, 0xAB, 0xDD, 0x95, 0xB5, 0xF3, 0x0D, 0x30, 0xA3, 0x29, 0xA7,
    0xED, 0x07, 0x68, 0xEB, 0x5F, 0x3A, 0x72, 0x21, 0xF9, 0xF2, 0xF6, 0xD3,
    0x74, 0x16, 0x0E, 0x39, 0x70, 0xD1, 0x0C, 0xF5, 0x5E, 0xE6, 0x76, 0x01,
    0x9D, 0x43, 0x4F, 0xE5, 0xE3, 0xD8, 0x98, 0x2D, 0xDF, 0xA9, 0x17, 0x1A,
    0x1D, 0x22, 0x57, 0x98, 0x29, 0xAB, 0xF0, 0x70, 0xF7, 0x7D, 0x08, 0xBF,
    0x82, 0xB4, 0x1A, 0xA7, 0xB4, 0xA5, 0xF7, 0x44, 0x86, 0x71, 0x0A, 0xE1,
    0x68, 0xDB, 0xDF, 0x6C, 0xE3, 0x56, 0xE7, 0x28, 0x7D, 0x62, 0x75, 0xA2,
    0xDD, 0x19, 0x0D, 0xE6, 0x2F, 0x05, 0xDE, 0xF2, 0xCA, 0x35, 0x8D, 0x61,
    0x10, 0x90, 0x34, 0x8F, 0x96, 0x69, 0xAB, 0x6B, 0x86, 0x53, 0x06, 0x4E,
    0x41, 0x36, 0x5E, 0x7D, 0x48, 0xA2, 0xD5, 0x28, 0x5C, 0x64, 0xD4, 0x9D,
    0xE4, 0x57, 0x97, 0xBC, 0x90, 0x7D, 0xEC, 0xAB, 0x3C, 0x4E, 0x37, 0xCB,
    0xB1, 0x1A, 0xBA, 0xEA, 0x04, 0xE8, 0x13, 0xBA, 0x40, 0x70, 0x06, 0x81,
    0x45, 0x89, 0x7D, 0x13, 0xEF, 0xB3, 0xCD, 0xFE, 0xDE, 0x1B, 0x3B, 0x04,
    0x1B, 0xB3, 0xBF, 0x00};

void xor_encryption(unsigned char* data, int length, const unsigned char* key, int key_length) {
  for (int i = 0; i < length; i++) {
    data[i] = data[i] ^ key[i % key_length];
  }
}

ModelInferRKNN::~ModelInferRKNN() { 
    rknn_destroy(rk_ctx_); 
    //for (int i=0; i<rk_io_num_.n_output; i++) {
    //  rknn_outputs_release(rk_ctx_, 1, &output_tensors_[i]);
    //}
    rknn_outputs_release(rk_ctx_, rk_io_num_.n_output, output_tensors_.data());
    if (model != nullptr) {
      free(model);
      model = nullptr;
    }
}

// encryption 0：load source model  1: load encryption model
unsigned char* LoadModelFromFile(std::string& model_name, int* model_size, bool encryption) {
  if (encryption) {
    FILE* fp = fopen(model_name.c_str(), "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", model_name.c_str());
        return NULL;
    }
    
    int model_len=0;
    fread(&model_len, sizeof(int), 1, fp);
    
    char author_new[100];
    fread(author_new, 100, 1, fp);
    //printf("load encryption model mdoel_len:%d, %s\n", model_len, author_new);

    if (strcmp(author_new, "hyfacejpyang81327261babffd0d6794a015652cfe48") == 0) {
      //printf("the same models\n");
      unsigned char* model_new = (unsigned char*)malloc(model_len);
      fread(model_new, model_len, 1, fp);
      xor_encryption(model_new, model_len, key, 256);
      if (fp) {
        fclose(fp);
      }
      *model_size = model_len;
      return model_new; 
    } else {
      return NULL;
    }

  } else {
    FILE *fp = fopen(model_name.c_str(), "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", model_name.c_str());
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char* model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp)) {
      printf("fread %s fail!\n", model_name.c_str());
      free(model);
      return NULL;
    }
    *model_size = model_len;
    if(fp) {
      fclose(fp);
    }
    return model;
  }
}

int ModelInferRKNN::InitRunner(const ModelConfig &config, ZipWrapper *wrapper) {
     #if AUTH_CHECK
	if (!init_call) {
      pthread_t   thread_id;    
      pthread_create (&thread_id, NULL, auth_check, NULL);
      do {
          sleep(2);
        }while(check_done == 0);
        init_call ++;
    }
    #endif
	
    run_status_ = false;
    if (config.infer_backend == "RKNPU") {
      config_ = config;
      std::vector<char> file_model = wrapper->ReadFileBinary(config.filename);
      int ret;
      ret = rknn_init(&rk_ctx_, file_model.data(), file_model.size(), 0);
      if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
      }
      ret = rknn_query(rk_ctx_, RKNN_QUERY_IN_OUT_NUM, &rk_io_num_,
                       sizeof(rk_io_num_));

      if (ret != RKNN_SUCC) {
        printf("rknn_query ctx fail! ret=%d\n", ret);
        return -1;
      }
      printf("model: %s, model input num: %d, output num: %d\n", config.filename.c_str(), rk_io_num_.n_input, rk_io_num_.n_output);
      //printf("input tensors:\n");
      input_attrs_.resize(rk_io_num_.n_input);
      output_attrs_.resize(rk_io_num_.n_output);
      input_tensors_.resize(rk_io_num_.n_input);
      output_tensors_.resize(rk_io_num_.n_output);

      for (int i = 0; i < config.input_nodes.size(); ++i) {
        memset(&input_attrs_[i], 0, sizeof(input_attrs_[i]));
        memset(&input_tensors_[i], 0, sizeof(input_tensors_[i]));
        input_attrs_[i].index = i;
        ret = rknn_query(rk_ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]),
                         sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
          printf("rknn_query fail! ret=%d\n", ret);
          return -1;
        }
      }

      for (int i = 0; i < config.output_nodes.size(); ++i) {
        memset(&output_attrs_[i], 0, sizeof(output_attrs_[i]));
        memset(&output_tensors_[i], 0, sizeof(output_tensors_[i]));
        output_attrs_[i].index = i;
        ret = rknn_query(rk_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]),
                         sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
          printf("rknn_query fail! ret=%d\n", ret);
          return -1;
        }
      }
    }
}

int ModelInferRKNN::InitRunner(const ModelConfig &config, std::string& modelfile_name) {
    #if AUTH_CHECK
	if (!init_call) {
      pthread_t   thread_id;    
      pthread_create (&thread_id, NULL, auth_check, NULL);
      do {
          sleep(2);
        }while(check_done == 0);
        init_call ++;
    }
    #endif
	run_status_ = false;
    if (config.infer_backend == "RKNPU") {
      config_ = config;

      // model_size model vector

      //load_model
      //std::vector<char> file_model = wrapper->ReadFileBinary(config.filename);
      int model_size=0;
      model = LoadModelFromFile(modelfile_name, &model_size, 1);

      //printf("LoadModelFromFile: %s, %d\n", modelfile_name.c_str(), model_size);

      int ret;
      ret = rknn_init(&rk_ctx_, model, model_size, 0);
      if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
      }
      ret = rknn_query(rk_ctx_, RKNN_QUERY_IN_OUT_NUM, &rk_io_num_,
                       sizeof(rk_io_num_));

      if (ret != RKNN_SUCC) {
        printf("rknn_query ctx fail! ret=%d\n", ret);
        return -1;
      }
      printf("model filename:%s, model input num: %d, output num: %d\n", config.filename.c_str(), rk_io_num_.n_input, rk_io_num_.n_output);
      //printf("input tensors:\n");
      input_attrs_.resize(rk_io_num_.n_input);
      output_attrs_.resize(rk_io_num_.n_output);
      input_tensors_.resize(rk_io_num_.n_input);
      output_tensors_.resize(rk_io_num_.n_output);

      for (int i = 0; i < config.input_nodes.size(); ++i) {
        memset(&input_attrs_[i], 0, sizeof(input_attrs_[i]));
        memset(&input_tensors_[i], 0, sizeof(input_tensors_[i]));
        input_attrs_[i].index = i;
        ret = rknn_query(rk_ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]),
                         sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
          printf("rknn_query fail! ret=%d\n", ret);
          return -1;
        }
      }

      for (int i = 0; i < config.output_nodes.size(); ++i) {
        memset(&output_attrs_[i], 0, sizeof(output_attrs_[i]));
        memset(&output_tensors_[i], 0, sizeof(output_tensors_[i]));
        output_attrs_[i].index = i;
        ret = rknn_query(rk_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]),
                         sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
          printf("rknn_query fail! ret=%d\n", ret);
          return -1;
        }
      }
    }
}

std::vector<int> ModelInferRKNN::GetInputTensorSize(const int &index) {
    std::vector<int> dims(input_attrs_[index].dims, input_attrs_[index].dims + input_attrs_[index].n_dims);
    return dims;
}

std::vector<int> ModelInferRKNN::GetOutputTensorSize(const int &index) {
      //std::cout<<"output_attrs_[index].n_dims:"<<output_attrs_[index].n_dims<<std::endl;
    std::vector<int> dims(output_attrs_[index].dims, output_attrs_[index].dims + output_attrs_[index].n_dims);
    return dims;
}

int ModelInferRKNN::GetOutputTensorLen(const int &index) {
      std::vector<int> tensor_size_out = GetOutputTensorSize(index);
      int size= 1;
      for(auto &one:tensor_size_out) size*=one;
      return size;
}

Status ModelInferRKNN::SetInputData(const int index, const cv::Mat &data) {
    if (data.type() != CV_8UC3) {
      printf("error: input data required CV_8UC3 \n");
    }
    if (index < input_tensors_.size()) {
      input_tensors_[index].index = 0;
      input_tensors_[index].type = RKNN_TENSOR_UINT8;
      input_tensors_[index].size = data.cols * data.rows * data.channels();
      input_tensors_[index].fmt = RKNN_TENSOR_NHWC;
      input_tensors_[index].buf = data.data;
    } else {
      printf("error: assert index < len\n");
    }

}

int ModelInferRKNN::RunModel() {
    //printf("set input\n");
    int ret = rknn_inputs_set(rk_ctx_, input_attrs_.size(), input_tensors_.data());
    if(ret < 0 )
        printf("rknn_run fail! ret=%d\n", ret);

    //printf("rknn_run\n");
    ret = rknn_run(rk_ctx_, nullptr);
    if (ret < 0) {
      printf("rknn_run fail! ret=%d\n", ret);
      return -1;
    }
    for(int i = 0 ; i < output_tensors_.size() ;++i){
        output_tensors_[i].want_float = 1;
    }
    ret = rknn_outputs_get(rk_ctx_,output_tensors_.size() , output_tensors_.data(), NULL);
    if (ret < 0) {
      printf("rknn_run fail! ret=%d\n", ret);
      exit(0);
    }
}

const float* ModelInferRKNN::GetOutputData(const int index) {
        return (float*)(output_tensors_[index].buf);
}

void ModelInferRKNN::ResizeInputTensor(const std::string &index_name, const std::vector<int> &shape) {
    // No implementation
}

void ModelInferRKNN::CheckSize() {
    // No implementation
}

