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
#include <stdio.h>
#include <memory.h>
#include <sys/time.h>
#include <string>
#include "helper.h"
#include "rockx.h"
#include "opencv2/opencv.hpp"
#include "face_landmark.h"
//#include "feature_extract_tool.h"
////#include "helper.h"
//#include "iostream"
//#include "opencv2/opencv.hpp"
//#include <sstream>

using namespace std;

double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int get_maxarea_face(rockx_object_array_t face_array) {
    int face_num   = face_array.count;
    int max_area   = 0;
    int max_index  = 0;
    for (int i=0; i<face_num; i++) {
        int area = (face_array.object[i].box.right-face_array.object[i].box.left)*(face_array.object[i].box.bottom-face_array.object[i].box.top);
        if (area > max_area) {
            max_index = i;
            max_area  = area;
        }
    }
    return max_index;
}

#if 0
std::vector<cv::Point2f> ApplyTransformToPoints(const std::vector<cv::Point2f> &points, const cv::Mat &matrix) {
  assert(matrix.rows == 2);
  assert(matrix.cols == 3);
  double m00 = matrix.at<double>(0, 0);
  double m01 = matrix.at<double>(0, 1);
  double m02 = matrix.at<double>(0, 2);
  double m10 = matrix.at<double>(1, 0);
  double m11 = matrix.at<double>(1, 1);
  double m12 = matrix.at<double>(1, 2);
  std::vector<cv::Point2f> out_points(points.size());
  assert(out_points.size() == points.size());
  for (int j = 0; j < points.size(); j++) {
    out_points[j].x = points[j].x * m00 + points[j].y * m01 + m02;
    out_points[j].y = points[j].x * m10 + points[j].y * m11 + m12;
  }
  return out_points;
}
#endif

int main(int argc, char** argv) {

    string images_folder_path = argv[1];
    string model_dir          = argv[2];

    rockx_ret_t ret;
    rockx_handle_t face_det_handle;
    // read image
    rockx_image_t input_image;
    // create rockx_face_array_t for store result
    rockx_object_array_t face_array;
    memset(&face_array, 0, sizeof(rockx_object_array_t));
    // create a face detection handle
    ret = rockx_create(&face_det_handle, ROCKX_MODULE_FACE_DETECTION, nullptr, 0);
    if (ret != ROCKX_RET_SUCCESS) {
        printf("init rockx module ROCKX_MODULE_FACE_DETECTION error %d\n", ret);
    }

    // quality and landmarks test handle
    ZipWrapper zipWrapper(model_dir);
    FaceLandmark facelandmark;
    facelandmark.Reset(zipWrapper);
    
    FILE* det_result = fopen("106landmark_result.txt", "w");

    if (check_folder(images_folder_path, false)) {
        std::cout << "images_folder_path: " << images_folder_path << std::endl;
    } else {
        std::cout << "images_folder_path folder does not exist." << std::endl;
        return 1;
    }

    // Read folder
    string suffix = "jpg";
    vector<string> file_names;
    vector<string> path_list = readFileListSuffix(images_folder_path.c_str(), suffix.c_str(), file_names, false);
    std::cout << "len path list" << path_list.size() << std::endl;
    int test_num = 0;
    for (int idx = 0; idx < path_list.size(); idx++) {
        std::string path_list_idx = path_list[idx];
        std::cout << path_list_idx << std::endl;
        rockx_image_read(path_list_idx.c_str(), &input_image, 1);
        cv::Mat img = cv::imread(path_list_idx.c_str());
        test_num += 1;
        printf("test num:%d, img size: rows:%d, cols:%d\n", test_num, img.rows, img.cols);
        double began = get_current_time();

        // detect face
        ret = rockx_face_detect(face_det_handle, &input_image, &face_array, nullptr);
        if (ret != ROCKX_RET_SUCCESS) {
            printf("rockx_face_detect error %d\n", ret);
            return -1;
        }
        double end   = get_current_time();
        int face_num = face_array.count;
        // fprintf(det_result, "%s,%d,%f,", file_names[idx].c_str(), face_num, (end-began));
        if (face_num <= 0) {
            fprintf(det_result, "%s,0,%f\n", file_names[idx].c_str(),(end-began));
            continue;
        } else {
            int max_index = get_maxarea_face(face_array);
            int x  = face_array.object[max_index].box.left;
            int y  = face_array.object[max_index].box.top;
            int w  = face_array.object[max_index].box.right-face_array.object[max_index].box.left;
            int h  = face_array.object[max_index].box.bottom-face_array.object[max_index].box.top;
           printf("detect face:x:%d, y:%d, w:%d, h:%d\n", x, y, w, h); 
            #if 0
            int cx = x + w / 2.0;
            int cy = y + h / 2.0;
            int length = std::max(w, h) * 1.5 / 2;
            int x1 = int(cx - length);
            int y1 = int(cy - length);
            int x2 = int(cx + length);
            int y2 = int(cy + length);
            int wi = x2 - x1;
            int hi = y2 - y1;

            if (x1 <= 0) {
                x2 = x2-x1+1;
                x1 = x1-x1+1;
            }
            if (y1 <= 0) {
                y2 = y2-y1+1;
                y1 = y1-y1+1;
            }
            if (x2 >= img.rows) {
                x2 = img.rows-1;
                x1 = img.rows-wi-1;
            }
            if (y2 >= img.cols) {
                y2 = img.cols-1;
                y1 = img.cols-hi-1;
            }
            float scale = 96.0/ wi;
            #endif

            cv::Mat affine = facelandmark.ComputeCropMatrix(cv::Rect2f(x, y, w, h));
            cv::Mat invert_affine;
            cv::invertAffineTransform(affine, invert_affine);
            cv::Mat crop_img;
            cv::warpAffine(img, crop_img, affine, cv::Size(112,112));
            printf("crop_img:rows:%d, cols:%d\n", crop_img.rows, crop_img.cols);
            //return 0;
            // crop image data
            //cv::Mat crop_img = img(cv::Rect(x1, y1, wi, hi)).clone();
            //cv::Mat resize_in;
            //cv::resize(crop_img, resize_in, cv::Size(96,96));
            
            //cv::imwrite("landmark_crop.jpg", crop_img);
	    began = get_current_time();
            facelandmark.Extract(crop_img);
            end = get_current_time();

            std::vector<cv::Point2f> points      = facelandmark.getPoints();
            std::vector<cv::Point2f> real_points = ApplyTransformToPoints(points, invert_affine);
            
            //fprintf(det_result, "%s,%d,%d,%f,%f,%f,%f,%f,%f,%f\n", file_names[idx].c_str(),angle,blur,(end-began),yaw,roll,pitch,quality_[0],quality_[1],quality_[2]);
            fprintf(det_result, "%s,1,%f,", file_names[idx].c_str(),(end-began));
            int shift = 3;
            int factor = (1 << shift);
            for (int i=0; i<points.size(); i++) {
                fprintf(det_result, "%f,%f,", real_points[i].x, real_points[i].y);
                cv::circle(img, cv::Point(real_points[i].x, real_points[i].y), 1, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
            }
            fprintf(det_result, "\n");
            cv::imwrite("106landmark.jpg", img);
            //return 0;
        }
        
	rockx_image_release(&input_image);
    }
    
    fclose(det_result);
    // release
    //rockx_image_release(&input_image);
    rockx_destroy(face_det_handle);
}
