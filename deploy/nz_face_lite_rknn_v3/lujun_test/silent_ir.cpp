/****************************************************************************
*
*    Copyright (c) 2017 - 2019 by Rockchip Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Rockchip Corporation. This is proprietary information owned by
*    Rockchip Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Rockchip Corporation.
*
*****************************************************************************/
#if 0
#include <stdio.h>
#include <memory.h>
#include <sys/time.h>
#include <string>
#include "helper.h"
//#include "rockx.h"
#include "opencv2/opencv.hpp"
#include "face_landmark.h"
#include "silent_live_ir.h"
//#include "feature_extract_tool.h"
////#include "helper.h"
//#include "iostream"
//#include "opencv2/opencv.hpp"
//#include <sstream>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include <string>
#include "helper.h"
#include "face_quality.h"
#include "silent_live_ir.h"
#include "feature_extract.h"
#include "live_rgb.h"
#include "face_landmark.h"
#include "iostream"
#include "opencv2/opencv.hpp"
#include <sstream>

using namespace std;

double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int get_maxarea_face(std::vector<BoxInfo>& boxes) {
    int face_num   = boxes.size();
    int max_area   = 0;
    int max_index  = 0;
    for (int i=0; i<face_num; i++) {
        int area = (boxes[i].x2-boxes[i].x1)*(boxes[i].y2-boxes[i].y1);
        if (area > max_area) {
            max_index = i;
            max_area  = area;
        }
    }
    return max_index;
}

int main(int argc, char** argv) {
  string images_folder_path = argv[1];
  string model_dir          = argv[2];
  string label              = argv[3];

  // face_detection alg
  RetinaFaceV2 face_detection;
  face_detection.Reset(model_dir);
 
  SilentLiveIR silent_ir;
  silent_ir.Reset(model_dir);
  
  // quality and 5 key points alg
  FaceQuality facequality;
  facequality.Reset(model_dir);

  //std::vector<BoxInfo> boxes = face_detection.Detect(img, 0.6, 0.3);
  FILE* det_result = fopen("result_ir.txt", "a");
  if (check_folder(images_folder_path, false)) {
      std::cout << "images_folder_path: " << images_folder_path << std::endl;
  } else {
      std::cout << "images_folder_path folder does not exist." << std::endl;
      return 1;
  }

  // Read folder
  string suffix = "png";
  vector<string> file_names;
  vector<string> path_list = readFileListSuffix(images_folder_path.c_str(), suffix.c_str(), file_names, false);
  std::cout << "len path list" << path_list.size() << std::endl;
  int test_num = 0;
  for (int idx = 0; idx < path_list.size(); idx++) {
        std::string path_list_idx = path_list[idx];
        std::cout << file_names[idx] << " " << label << " ";
        fprintf(det_result, "%s %s ", file_names[idx].c_str(), label.c_str());
        cv::Mat img = cv::imread(path_list_idx.c_str());
        test_num += 1;
        double b = get_current_time();
        std::vector<BoxInfo> boxes = face_detection.Detect(img, 0.6, 0.3);
        double e = get_current_time();
        int x=0,y=0,w=0,h=0;
        int face_num = boxes.size();
        if (face_num <= 0) {
          // fprintf(det_result, "%s,0,%f\n", file_names[idx].c_str(),(end-began));
          std::cout << "-1 0.0 0.0 " << (e-b) << std::endl;
          fprintf(det_result, "-1 0.0 0.0 %f\n", (e-b));
        } else {
          int max_index = get_maxarea_face(boxes);
          x  = boxes[max_index].x1;
          y  = boxes[max_index].y1;
          w  = boxes[max_index].x2 - boxes[max_index].x1;
          h  = boxes[max_index].y2 - boxes[max_index].y1;
          double b = get_current_time();
          facequality.Extract(img, cv::Rect2f(x, y , w, h));
          double e   = get_current_time();
          // printf("quality alg need time:%f\n", e-b);                    
          float yaw                            = facequality.getYaw();
          float roll                           = facequality.getRoll();
          float pitch                          = facequality.getPitch();
          std::vector<float> quality_          = facequality.getQuality();
          std::vector<cv::Point2f> points      = facequality.getPoints();
          std::vector<cv::Point2f> real_points = facequality.getSourceImagePoints();
          // printf("face angle and quality: %f, %f, %f, %f, %f, %f\n", yaw, roll, pitch, quality_[0], quality_[1], quality_[2]);
          std::string outName = images_folder_path + "_crop/" + file_names[idx];
          double b1 = get_current_time();
          float live_score = silent_ir.Check(img, real_points, outName);
          double e1 = get_current_time();
          // printf("live score:%f, need time:%f ms\n", live_score, (e-b));
          int out = live_score >= 0.25 ? 0 : 1;
          std::cout << out << " " << live_score << " " << (e1-b1) << " " << (e-b) <<std::endl;    
          fprintf(det_result, "%d %f %f %f\n", out, live_score, (e1-b1), (e-b));    
        } 
    }
    fclose(det_result);
  //float live_score = rgblive.Check(img);  
  //printf("live score:%f\n", live_score);
}

// #if 0
// int main(int argc, char** argv) {

//     string images_folder_path = argv[1];
//     string model_dir          = argv[2];

//     // face_detection alg
//     RetinaFaceV2 face_detection;
//     face_detection.Reset(model_dir);
    
//     // ir_alvie alg
//     SilentLiveIR slient_ir;
//     slient_ir.Reset(model_dir);
	
//     FILE* det_result = fopen("alvie_result.txt", "w");

//     if (check_folder(images_folder_path, false)) {
//         std::cout << "images_folder_path: " << images_folder_path << std::endl;
//     } else {
//         std::cout << "images_folder_path folder does not exist." << std::endl;
//         return 1;
//     }

//     // Read folder
//     string suffix = "jpg";
//     vector<string> file_names;
//     vector<string> path_list = readFileListSuffix(images_folder_path.c_str(), suffix.c_str(), file_names, false);
//     std::cout << "len path list" << path_list.size() << std::endl;
//     int test_num = 0;
//     for (int idx = 0; idx < path_list.size(); idx++) {
//         std::string path_list_idx = path_list[idx];
//         std::cout << path_list_idx << std::endl;
//         cv::Mat img = cv::imread(path_list_idx.c_str());
//         test_num += 1;

// 		double began = get_current_time();
//         // detect face
// 		std::vector<BoxInfo> boxes = face_detection.Detect(img, 0.6, 0.3);
//         double end   = get_current_time();
//         // fprintf(det_result, "%s,%d,%f,", file_names[idx].c_str(), face_num, (end-began));
// 		// get maxarea
//         int x=0,y=0,w=0,h=0;
//         int face_num = boxes.size();
//         if (face_num <= 0) {
//             fprintf(det_result, "%s,0,%f\n", file_names[idx].c_str(),(end-began));
//             continue;
//         } else {
//             int max_index = get_maxarea_face(boxes);
//             x  = boxes[max_index].x1;
//             y  = boxes[max_index].y1;
//             w  = boxes[max_index].x2 - boxes[max_index].x1;
//             h  = boxes[max_index].y2 - boxes[max_index].y1;
//             printf("detect face:x:%d, y:%d, w:%d, h:%d\n", x, y, w, h);
//             fprintf(det_result, "%s,1,face area: x:%d, y:%d, x2:%d, y2:%d\n", file_names[idx].c_str(), x, y, x+w, y+h);
			
// 			//---------------------------------------------
// 			// step 6:  alive  
// 			//---------------------------------------------
// 			float score = slient_ir.Check(img, face_detection, cv::Rect2f(x, y , w, h), 0.08f, 0.07f, 0.f, 0.f);
// 			fprintf(det_result, "%s,quality_pass,score:%f\n", file_names[idx].c_str(),score);
//         }
//     }    
//     fclose(det_result);
// 	return 0;
// }
// #endif

