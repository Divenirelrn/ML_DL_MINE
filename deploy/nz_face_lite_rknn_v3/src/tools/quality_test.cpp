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
#include "face_quality.h"
//#include "feature_extract_tool.h"
//#include "helper.h"
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

int main(int argc, char** argv) {
    
    string model_dir = argv[1];
    string images_folder_path = argv[2];

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
    FaceQuality facequality;
    facequality.Reset(zipWrapper);
    
    FILE* det_result = fopen("quality_no_rota_result.txt", "w");

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
            continue;
        } else {
            int max_index = get_maxarea_face(face_array);
            int x  = face_array.object[max_index].box.left;
            int y  = face_array.object[max_index].box.top;
            int w  = face_array.object[max_index].box.right-face_array.object[max_index].box.left;
            int h  = face_array.object[max_index].box.bottom-face_array.object[max_index].box.top;
            
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
            assert(wi==hi);
	    float scale = 96.0 / wi;
            printf("padbefore:x1:%d,y1:%d,x2:%d,y2:%d,w:%d,h:%d\n", x1, y1, x2, y2, wi, hi);
            if (x1 <= 0) {
                x2 = x2-x1+1;
                x1 = 1;
            }
            if (y1 <= 0) {
                y2 = y2-y1+1;
                y1 = 1;
            }
            if (x2 >= img.rows) {
                x2 = img.rows-1;
                x1 = img.rows-wi-1;
            }
            if (y2 >= img.cols) {
                y2 = img.cols-1;
                y1 = img.cols-hi-1;
            }
	    printf("padafter:x1:%d,y1:%d,x2:%d,y2:%d,w:%d,h:%d\n", x1, y1, x2, y2, wi, hi);
#endif
	    cv::Mat affine = facequality.ComputeCropMatrix(cv::Rect2f(x, y, w, h));
	    cv::Mat invert_affine;
	    cv::invertAffineTransform(affine, invert_affine);
	    
	    cv::Mat crop_img;
	    cv::warpAffine(img, crop_img, affine, cv::Size(96,96));
	    cv::imwrite("warper.jpg", crop_img);
	    printf("crop_img:rows:%d, cols:%d\n", crop_img.rows, crop_img.cols);
            //return 0;

            //wi = x2-x1;
	    //hi = y2-y1;
	    //assert(wi==hi);
	    //float = 
            // crop image data
            //cv::Mat crop_img = img(cv::Rect(x1, y1, wi, hi)).clone();
            //cv::Mat crop_img = img(cv::Rect(x1, y1, wi, hi)).clone();
	    //cv::Mat resize_in;
	    //cv::resize(crop_img, resize_in, cv::Size(96,96));

            began = get_current_time();
	    facequality.Extract(crop_img);
            end = get_current_time();
            float yaw                  = facequality.getYaw();
	    float roll                 = facequality.getRoll();
	    float pitch                = facequality.getPitch();
	    std::vector<float> quality_ = facequality.getQuality();						           
	    int angle = 1;  // 0 hege, 1:buheg
	    int blur  = 1;
 	    if (abs(yaw) <= 30. && abs(roll) <= 30. && abs(pitch) <= 30.) {											
		    angle = 0; 
	    }
	    if ((quality_[0]+quality_[1]+quality_[2])/3 <= 0.45) { 													 
		    blur = 0;
	    }
	    printf("%s,%f,%f,%f,%f,%f,%f\n",file_names[idx].c_str(), yaw,roll,pitch,quality_[0],quality_[1],quality_[2]);
            fprintf(det_result, "%s,%d,%d,%f,%f,%f,%f,%f,%f,%f\n", file_names[idx].c_str(),angle,blur,(end-began),yaw,roll,pitch,quality_[0],quality_[1],quality_[2]);

	    std::vector<cv::Point2f> points      = facequality.getPoints();
	    std::vector<cv::Point2f> real_points = ApplyTransformToPoints(points, invert_affine);
	    int shift = 3;
	    int factor = (1 << shift);
	    for (int i=0; i<points.size(); i++) {
	        cv::circle(img, cv::Point(real_points[i].x, real_points[i].y), 1, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
	    }														            
            //cv::imwrite("landmark.jpg", img);
	    //fprintf(det_result, "%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", file_names[idx].c_str(),(end-began),real_points[0].x, real_points[0].y, real_points[1].x, real_points[1].y, real_points[2].x, real_points[2].y, real_points[3].x, real_points[3].y, real_points[4].x, real_points[4].y);
           //return 0;
	}
        
	rockx_image_release(&input_image);
    }
    
    fclose(det_result);
    // release
    //rockx_image_release(&input_image);
    rockx_destroy(face_det_handle);
}
