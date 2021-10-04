#ifndef MODEL_INFER_H
#define MODEL_INFER_H

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

#if 0
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
        
        strncpy(begin_time, secret+(strlen(secret)-21), 10);
        strncpy(end_time,   secret+(strlen(secret)-11), 10);
        
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

class ModelInferRKNN {
public:
  ModelInferRKNN(const ModelInferRKNN &) = delete;
  ModelInferRKNN &operator=(const ModelInferRKNN &) = delete;
  ModelInferRKNN() {}
  ~ModelInferRKNN();
  int InitRunner(const ModelConfig &config, ZipWrapper *wrapper);
  int InitRunner(const ModelConfig &config, std::string& modelfile_name);
  ModelType GetModelType() { return MODEL_RKNN; }
  std::vector<int> GetInputTensorSize(const int &index);
  std::vector<int> GetOutputTensorSize(const int &index);
  int GetOutputTensorLen(const int &index);
  Status SetInputData(const int index, const cv::Mat &data);
  int RunModel();
  const float *GetOutputData(const int index);
  void ResizeInputTensor(const std::string &index_name, const std::vector<int> &shape);
  void CheckSize();

private:
  rknn_context rk_ctx_;
  rknn_input_output_num rk_io_num_;
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;

  std::vector<rknn_input> input_tensors_;
  std::vector<rknn_output> output_tensors_;
  std::vector<int> tensor_shape_;
  int width_;
  int height_;
  ModelConfig config_;
  bool run_status_;

  int model_size;
  unsigned char* model;
};

#endif // NCNN_MOBILEFACENET_FACE_LANDMARK_H
