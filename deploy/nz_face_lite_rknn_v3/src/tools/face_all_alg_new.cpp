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
    
    string model_dir          = argv[1];
    string image_rgb_path     = argv[2];
    string image_ir_path      = argv[3];
    int    loop_times         = atoi(argv[4]);

    //---------------------------------------------
    // step 1 : init (face_detection) (ir_alive) (quality_5_landmark) (106_landmark) (face feature) model
    //---------------------------------------------

    double b = get_current_time();
    // face_detection alg
    RetinaFaceV2 face_detection;
    face_detection.Reset(model_dir);
    
    // ir_alvie alg
    SilentLiveIR slient_ir;
    slient_ir.Reset(model_dir);
    
    // quality and 5 key points alg
    FaceQuality facequality;
    facequality.Reset(model_dir);
    
    // 106 landmark alg
    FaceLandmark facelandmark_106;
    facelandmark_106.Reset(model_dir);
    
    // face feature alg
    FeatureExtract featureExtract;
    featureExtract.Reset(model_dir);

    double e = get_current_time();
    printf("step 1 init self face detection, irlive, quality, landmark, features alg, need time:%f ms\n", (e-b));
    
    for (int i=0; i<loop_times; i++) {
        //---------------------------------------------
        // step 2: read image data
        //---------------------------------------------
        cv::Mat img      = cv::imread(image_rgb_path.c_str());
        cv::Mat img_show = img.clone();
        printf("step 2 read image data for alg\n");

        //---------------------------------------------
        // step 3: detect rgb data
        //---------------------------------------------
        double began               = get_current_time();
        std::vector<BoxInfo> boxes = face_detection.Detect(img, 0.6, 0.3);
        double end                 = get_current_time();
        printf("step 3 detect face num:%d need time:%f\n", boxes.size(), end-began);
        
        // get maxarea
        int x=0,y=0,w=0,h=0;
        int face_num = boxes.size();
        if (face_num <= 0) {
            continue;
        } else {
            int max_index = get_maxarea_face(boxes);
            x  = boxes[max_index].x1;
            y  = boxes[max_index].y1;
            w  = boxes[max_index].x2 - boxes[max_index].x1;
            h  = boxes[max_index].y2 - boxes[max_index].y1;
            cv::rectangle(img_show, cv::Rect(x,y,w,h), cv::Scalar(0, 0, 255), 1, 8, 0);
        }

        //---------------------------------------------
        // step 4:  quality and 5 landmarks
        //---------------------------------------------
        began = get_current_time();
        facequality.Extract(img, cv::Rect2f(x, y , w, h));
        end   = get_current_time();
        printf("step 4 quality alg need time:%f\n", end-began);
        
        float yaw                            = facequality.getYaw();
        float roll                           = facequality.getRoll();
        float pitch                          = facequality.getPitch();
        std::vector<float> quality_          = facequality.getQuality();
        std::vector<cv::Point2f> points      = facequality.getPoints();
        std::vector<cv::Point2f> real_points = facequality.getSourceImagePoints();
        printf("face angle and quality: %f, %f, %f, %f, %f, %f\n", yaw, roll, pitch, quality_[0], quality_[1], quality_[2]);

        if (abs(yaw) <= 80. && abs(roll) <= 80. && abs(pitch) <= 80. 
           && quality_[0] <= 0.9 && quality_[1] <= 0.9 && quality_[2] <= 0.9)
        {
            //---------------------------------------------
            // step 5: 106 landmark detect 
            //---------------------------------------------
            began = get_current_time();
            facelandmark_106.Extract(img, cv::Rect2f(x, y , w, h));
            end = get_current_time();
            printf("step 5 landmark 106 alg need time:%f\n", end-began);

            std::vector<cv::Point2f> real_points_land106 = facelandmark_106.getSourceImagePoints();
            for (int i=0; i<real_points_land106.size(); i++) {
                cv::circle(img_show, cv::Point(real_points_land106[i].x, real_points_land106[i].y), 1, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
            }
            cv::imwrite("rect_landmarks.jpg", img_show);

            //---------------------------------------------
            // step 6:  alive  
            //---------------------------------------------
            cv::Mat img_ir = cv::imread(image_ir_path.c_str());

            began = get_current_time();
            float score = slient_ir.Check(img_ir, face_detection, cv::Rect2f(x, y , w, h), 0.08f, 0.07f, 0.f, 0.f);
            end   = get_current_time();
            printf("step 6-1 crop ir detect ir image and check need time:%f, score:%f\n", end-began, score);


            began = get_current_time();
            float score_aligned = slient_ir.Check(img_ir, face_detection, facequality, cv::Rect2f(x, y , w, h), 0.08f, 0.07f, 0.f, 0.f);
            end   = get_current_time();
            printf("step 6-2 aligened ir detect  image and check need time:%f, score:%f\n", end-began, score_aligned);
            
	    cv::Mat crop_img = cv::imread("./ir_api_algined_self.png");
	    began = get_current_time();
	    float crop_score = slient_ir.Check(crop_img);
	    end   = get_current_time();

	    cv::Mat crop_img_flip;
	    cv::flip(crop_img, crop_img_flip, 1);
	    float crop_score_flip = slient_ir.Check(crop_img_flip);
	    printf("step 6-3 crop img ir detect  image and check need time:%f, score:%f, flip_score:%f, av:%f\n", end-began, crop_score, crop_score_flip, (crop_score+crop_score_flip)/2);

            if (score >= 0.001) {
                //---------------------------------------------
                // step 7:  recognition
                //---------------------------------------------
                began = get_current_time();
                float feat[512];
                std::vector<float> features = featureExtract.Extract(img, real_points, feat);
                end = get_current_time();
                float similar_score = featureExtract.GetSimilarScore(feat, feat);
                printf("step 7 face recognition alg need time:%f, similar_score:%f\n", end-began, similar_score);

            } else {
                printf("not alive face, please check\n");
            }
        } else {
            printf("image face angle or quality do not pass, please check\n");
        }
    }
    return 0;
}
