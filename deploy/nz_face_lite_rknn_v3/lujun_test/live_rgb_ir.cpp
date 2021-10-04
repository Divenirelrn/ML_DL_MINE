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
#include "retinafacev2.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include <string>
#include "helper.h"
#include "face_quality.h"
#include "silent_live_ir.h"
#include "live_rgb.h"
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

    string images_folder_path = argv[1];
    string model_dir          = argv[2];
    string label              = argv[3];

    RetinaFaceV2 face_detection;
    face_detection.Reset(model_dir);
    
    // ir_alvie alg
    SilentLiveIR slient_ir;
    slient_ir.Reset(model_dir);
    
    // rgb live
    RgbLive rgblive;
    rgblive.Reset(model_dir);
    
    // quality and 5 key points alg
    FaceQuality facequality;
    facequality.Reset(model_dir);
    
    // 106 landmark alg
    FaceLandmark facelandmark_106;
    facelandmark_106.Reset(model_dir);
    
    // face feature alg
    FeatureExtract featureExtract;
    featureExtract.Reset(model_dir);	

    FILE* pair_rgb_box = fopen("pair_rgb_box.txt", "a");
    FILE* pair_ir_box = fopen("pair_ir_box.txt", "a");

    // if (check_folder(images_folder_path, false)) {
    //     std::cout << "images_folder_path: " << images_folder_path << std::endl;
    // } else {
    //     std::cout << "images_folder_path folder does not exist." << std::endl;
    //     return 1;
    // }

    // Read folder
    string images_folder_rgb = images_folder_path + "_rgb";    
    string images_folder_ir = images_folder_path + "_ir";    
    string suffix = "png";
    vector<string> file_names;
    vector<string> path_list = readFileListSuffix(images_folder_rgb.c_str(), suffix.c_str(), file_names, false);
    std::cout << "len path list" << path_list.size() << std::endl;
    int test_num = 0;
	int x=0,y=0,w=0,h=0,face_num=0,max_index=0;
    for (int idx = 0; idx < path_list.size(); idx++) {
        std::string path_list_idx = path_list[idx];
        // std::cout << path_list_idx << std::endl;
        std::cout << file_names[idx] << " " << label << " ";
        fprintf(pair_rgb_box, "%s %s ", file_names[idx].c_str(), label.c_str());
        fprintf(pair_ir_box, "%s %s ", file_names[idx].c_str(), label.c_str());
        cv::Mat img = cv::imread(path_list_idx.c_str());
        test_num += 1;        
		
        // detect face rgb
        double b = get_current_time();
		std::vector<BoxInfo> boxes = face_detection.Detect(img, 0.6, 0.3);
        double e   = get_current_time();
		// get maxarea
        x=y=w=h=0;
        face_num = boxes.size();
        if (face_num <= 0) {
            std::cout << "-1 0.0 0.0 " << (e-b) << std::endl;
            fprintf(pair_rgb_box, "-1 0.0 0.0 %f\n", (e-b));
            fprintf(pair_ir_box, "-1 0.0 0.0 %f\n", (e-b));
            continue;
        } else {
            max_index = get_maxarea_face(boxes);
            x  = boxes[max_index].x1;
            y  = boxes[max_index].y1;
            w  = boxes[max_index].x2 - boxes[max_index].x1;
            h  = boxes[max_index].y2 - boxes[max_index].y1;
            std::string outName = images_folder_rgb + "_crop_box_pair/" + file_names[idx];
            double began = get_current_time();
            float score_rgb = rgblive.Check(img, cv::Rect(x, y, w, h), outName);
            double end = get_current_time(); 
            // if (fabs(live_score-MOVE_MIDDLE_SCREEN) <= 0.001){
            if (score_rgb < 0){
                // float score_tmp = fabs(live_score-MOVE_MIDDLE_SCREEN);
                std::cout << "-2 0.0 "<< (end-began) << " " << (e-b) << std::endl;
                fprintf(pair_rgb_box, "-2 0.0 %f %f\n", (end-began), (e-b));
            }else {
                // std::cout << live_score << " " << (end-began) << std::endl;
                int out = score_rgb >= 0.91 ? 0 : 1;
                std::cout << out << " " << score_rgb << " " << (end-began) << " " << (e-b) <<std::endl; 
                fprintf(pair_rgb_box, "%d %f %f %f\n", out, score_rgb, (end-began), (e-b));
            }
            
            string img_ir_path = images_folder_ir + "/" +  file_names[idx];
            cv::Mat img_ir = cv::imread(img_ir_path.c_str());
            std::string outName_ir = images_folder_ir + "_crop_box_pair/" + file_names[idx];
            double began_ir = get_current_time();
            float score_ir = slient_ir.Check(img, face_detection, cv::Rect(x, y, w, h), 0.08f, 0.07f, 0.0f, 0.0f, outName_ir);  
            double end_ir = get_current_time();
            if (score_ir < 0){
                std::cout << "-1 0.0 "<< (end_ir-began_ir) << " " << (e-b) << std::endl;
                fprintf(pair_ir_box, "-1 0.0 %f %f\n", (end_ir-began_ir), (e-b));
            }else {
                int out = score_ir >= 0.25 ? 0 : 1;
                std::cout << out << " " << score_ir << " " << (end_ir-began_ir) << " " << (e-b) <<std::endl; 
                fprintf(pair_ir_box, "%d %f %f %f\n", out, score_ir, (end_ir-began_ir), (e-b));
            }
        }
    }    
    fclose(pair_rgb_box);
    fclose(pair_ir_box);
	return 0;
}
