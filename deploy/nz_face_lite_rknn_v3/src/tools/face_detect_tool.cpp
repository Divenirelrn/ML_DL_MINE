//
// Created by YH-Mac on 2020/9/11.
//
#include "retinafacev2.h"
#include "helper.h"
#include "iostream"
#include "opencv2/opencv.hpp"
#include <sstream>
#include <string>
#include <sys/time.h>

double get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

using namespace std;


void print_feature(const std::string &filename, std::vector<float> &feature) {
    int feature_len = feature.size();
    std::cout << filename << ":";
    for (int i = 0; i < feature_len; i++) {
        std::cout << feature[i] << " ";
    }
    std::cout << std::endl;
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

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cout << argv[0] << " ";
        std::cout << "model_path pic_path" << std::endl;
        return 0;
    }

    FILE* det_result = fopen("retinafacev2_final_256_result.txt", "w");

    string model_dir = argv[1];
    string images_folder_path = argv[2];
    //string zip_dir   = argv[3]; 

    //double b = get_current_time();
    //ZipWrapper zipWrapper_zip(zip_dir);
    //double e = get_current_time();
    //printf("unzip zip model need time:%f ms\n", (e-b));

    //b = get_current_time();
    //ZipWrapper zipWrapper(model_dir);
    //e = get_current_time();
    //printf("unzip model file need time:%f ms\n", (e-b));
    
    RetinaFaceV2 retinaface_v2;
    double b = get_current_time();
    retinaface_v2.Reset(model_dir);
    double e = get_current_time();
    printf("resnet det model need time:%f ms\n", e-b);

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
        cv::Mat img = cv::imread(path_list_idx.c_str());
        b = get_current_time();
        std::vector<BoxInfo> boxes = retinaface_v2.Detect(img,0.5,0.3);
        e= get_current_time();
        printf("detect need time:%f ms\n", e-b);
        test_num += 1;
        printf("test num:%d, img size: rows:%d, cols:%d\n", test_num, img.rows, img.cols);
        
        int face_num = boxes.size();
        if (face_num <= 0) {
            fprintf(det_result, "%s,0,%f\n", file_names[idx].c_str(),(e-b));
            continue;
        } else {
            fprintf(det_result, "%s,%d,%f,", file_names[idx].c_str(),face_num,(e-b));
            for (auto bbox:boxes)
            {
                int x1 = static_cast<int>(bbox.x1);
                int y1 = static_cast<int>(bbox.y1);
                int x2 = static_cast<int>(bbox.x2);
                int y2 = static_cast<int>(bbox.y2);
                printf("%d %d %d %d \n" , x1,y1,x2,y2);
                fprintf(det_result, "%d,%d,%d,%d,%f,", x1,y1,x2,y2,bbox.score);
            }
            fprintf(det_result, "\n");
        }
    }
    fclose(det_result);
    return 0;
}
