//======================================================================================
//
//        Copyright (C) 2021 THL A29 Limited, a hangyi company. All rights reserved.
//
//        filename :face_all_alg.hpp
//        description : all face alg test on rv1109
//        created by yangjunpei at  15/05/2021 18:19:28
//
//======================================================================================
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include <string>
#include "helper.h"
#include "rockx.h"
#include "face_quality.h"
#include "silent_live_ir.h"
#include "feature_extract.h"
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
    
    string model_dir          = argv[1];
    string image_rgb_path     = argv[2];
    string image_ir_path      = argv[3];
    int    loop_times         = atoi(argv[4]);

    //---------------------------------------------
    // step 1 : init rockx detection
    // --------------------------------------------
    // rockx detection need use
    rockx_ret_t ret;
    rockx_handle_t face_det_handle;
    
    // read image, not forget free
    rockx_image_t input_image;

    // create rockx_face_array_t for store result for rgb detect result use
    rockx_object_array_t face_array;
    memset(&face_array, 0, sizeof(rockx_object_array_t));

    // create rockx_face_array_t for store result for ir detect result use
    rockx_object_array_t face_array_ir;
    memset(&face_array_ir, 0, sizeof(rockx_object_array_t));

    // create a face detection handle
    ret = rockx_create(&face_det_handle, ROCKX_MODULE_FACE_DETECTION, nullptr, 0);
    if (ret != ROCKX_RET_SUCCESS) {
        printf("init rockx module ROCKX_MODULE_FACE_DETECTION error %d\n", ret);
    }
    printf("step 1 init rockx detection alg\n");

    //---------------------------------------------
    // step 2 : init (ir_alive) (quality_5_landmark) (106_landmark) (face feature) model
    //---------------------------------------------
    ZipWrapper zipWrapper(model_dir);
    // ir_alvie alg
    SilentLiveIR slient_ir;
    slient_ir.Reset(zipWrapper);
    // quality and 5 key points alg
    FaceQuality facequality;
    facequality.Reset(zipWrapper);
    // 106 landmark alg
    FaceLandmark facelandmark_106;
    facelandmark_106.Reset(zipWrapper);
    // face feature alg
    FeatureExtract featureExtract;
    featureExtract.Reset(zipWrapper);
    printf("step 2 init self irlive, quality, landmark, features alg\n");
    

    for (int i=0; i<loop_times; i++) {
        //---------------------------------------------
        // step 3: read image data
        //---------------------------------------------
        rockx_image_read(image_rgb_path.c_str(), &input_image, 1);
        cv::Mat img      = cv::imread(image_rgb_path.c_str());
        cv::Mat img_show = img.clone();
        printf("step 3 read image data for alg\n");

        //---------------------------------------------
        // step 4: detect rgb data
        //---------------------------------------------
        double began = get_current_time();
        ret = rockx_face_detect(face_det_handle, &input_image, &face_array, nullptr);
        if (ret != ROCKX_RET_SUCCESS) {
            printf("rockx_face_detect error %d\n", ret);
            return -1;
        }
        double end   = get_current_time();
        printf("step 4 detect face num:%d need time:%f\n", face_array.count, end-began);
        // need free the image data
        rockx_image_release(&input_image);

        // get maxarea
        int x=0,y=0,w=0,h=0;
        int face_num = face_array.count;
        if (face_num <= 0) {
            continue;
        } else {
            int max_index = get_maxarea_face(face_array);
            x  = face_array.object[max_index].box.left;
            y  = face_array.object[max_index].box.top;
            w  = face_array.object[max_index].box.right-face_array.object[max_index].box.left;
            h  = face_array.object[max_index].box.bottom-face_array.object[max_index].box.top;

            cv::rectangle(img_show, cv::Rect(x,y,w,h), cv::Scalar(0, 0, 255), 1, 8, 0);
        }

        //---------------------------------------------
        // step 5:  quality and 5 landmarks
        //---------------------------------------------
        cv::Mat affine = facequality.ComputeCropMatrix(cv::Rect2f(x, y, w, h));
        cv::Mat invert_affine;
        cv::invertAffineTransform(affine, invert_affine);
        cv::Mat crop_img;
        cv::warpAffine(img, crop_img, affine, cv::Size(96,96));
        
        began = get_current_time();
        facequality.Extract(crop_img);
        end   = get_current_time();
        printf("step 5 quality alg need time:%f\n", end-began);
        
        float yaw                            = facequality.getYaw();
        float roll                           = facequality.getRoll();
        float pitch                          = facequality.getPitch();
        std::vector<float> quality_          = facequality.getQuality();
        std::vector<cv::Point2f> points      = facequality.getPoints();
        std::vector<cv::Point2f> real_points = ApplyTransformToPoints(points, invert_affine);
        printf("face angle and quality: %f, %f, %f, %f, %f, %f\n", yaw, roll, pitch, quality_[0], quality_[1], quality_[2]);

        if (abs(yaw) <= 30. && abs(roll) <= 30. && abs(pitch) <= 30. 
           && quality_[0] <= 0.45 && quality_[1] <= 0.45 && quality_[2] <= 0.45)
        {
            //---------------------------------------------
            // step 6: 106 landmark detect 
            //---------------------------------------------
            cv::Mat affine_land106 = facelandmark_106.ComputeCropMatrix(cv::Rect2f(x, y, w, h));
            cv::Mat invert_affine_land106;
            cv::invertAffineTransform(affine_land106, invert_affine_land106);
            cv::Mat crop_img_land106;
            cv::warpAffine(img, crop_img_land106, affine_land106, cv::Size(112,112));
            began = get_current_time();
            facelandmark_106.Extract(crop_img_land106);
            end = get_current_time();
            printf("step 6 landmark 106 alg need time:%f\n", end-began);

            std::vector<cv::Point2f> points_land106      = facelandmark_106.getPoints();
            std::vector<cv::Point2f> real_points_land106 = ApplyTransformToPoints(points_land106, invert_affine_land106);

            for (int i=0; i<real_points_land106.size(); i++) {
                cv::circle(img_show, cv::Point(real_points_land106[i].x, real_points_land106[i].y), 1, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
            }
            cv::imwrite("rect_landmarks.jpg", img_show);

            //---------------------------------------------
            // step 7:  alive  
            //---------------------------------------------
            rockx_image_read(image_ir_path.c_str(), &input_image, 1);
            cv::Mat img_ir = cv::imread(image_ir_path.c_str());

            began = get_current_time();
            ret = rockx_face_detect(face_det_handle, &input_image, &face_array_ir, nullptr);
            if (ret != ROCKX_RET_SUCCESS) {
                printf("rockx_face_detect error %d\n", ret);
                return -1;
            }
            end   = get_current_time();
            printf("step 7 detect ir image need time:%f\n", end-began);
            // need free the image data
            rockx_image_release(&input_image);
            int face_num_ir = face_array_ir.count;
            int x_=0,y_=0,w_=0,h_=0;
            if (face_num_ir <= 0) {
                continue;
            } else {
                int max_index  = get_maxarea_face(face_array_ir);
                x_         = face_array_ir.object[max_index].box.left;
                y_         = face_array_ir.object[max_index].box.top;
                w_         = face_array_ir.object[max_index].box.right-face_array_ir.object[max_index].box.left;
                h_         = face_array_ir.object[max_index].box.bottom-face_array_ir.object[max_index].box.top;
            }

            cv::Mat affine_alive = slient_ir.ComputeCropMatrix(cv::Rect2f(x_, y_, w_, h_));
            cv::Mat crop_img_alive;
            cv::warpAffine(img_ir, crop_img_alive, affine_alive, cv::Size(96,96));
	    cv::imwrite("crop_ir.jpg", crop_img_alive);
            began = get_current_time();
            float score = slient_ir.Check(crop_img_alive);
            end = get_current_time();
            printf("step 7 face alive alg score:%f, need time:%f\n", score, end-began);

            if (score >= 0.001) {
                //---------------------------------------------
                // step 8:  recognition
                //---------------------------------------------
                float lds[10] = {0,};
                for (int j=0; j<real_points.size(); j++) {
                    lds[2*j]    = real_points[j].x;
                    lds[2*j + 1] = real_points[j].y;
                }
                cv::Mat detlds(5, 2, CV_32F);
                detlds.data = (unsigned char *)lds;
                cv::Mat algined;
                featureExtract.Alignment(img, algined, cv::Rect(2,3,10,10), detlds);
                cv::imwrite("algined_img.jpg", algined);
                began = get_current_time();
                std::vector<float> features = featureExtract.Extract(algined);
                end = get_current_time();
                printf("step 8 face recognition alg need time:%f\n", end-began);

            } else {
                printf("not alive face, please check\n");
            }
        } else {
            printf("image face angle or quality do not pass, please check\n");
        }
    }
    rockx_destroy(face_det_handle);
}
