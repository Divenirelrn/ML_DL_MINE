#include "mobilenet.h"
#include "helper.h"
#include <iostream>
#include <string>
#include <map>

int main(int argc, char **argv){
    //Image filder & model dir
    std::string images_folder_path = argv[1];
    std::string model_dir          = argv[2];

    //Model
    MobileNet model;
    std::cout << "Model has initialized!" << std::endl;
    model.Reset(model_dir);

    //Image folder verify
	if (check_folder(images_folder_path, false)) {
        std::cout << "images_folder_path: " << images_folder_path << std::endl;
    } else {
        std::cout << "images_folder_path folder does not exist." << std::endl;
        return 1;
    }

    //Map of class index & name
	std::map<int, std::string> mp;
	mp.insert(std::pair<int,std::string>(0,"fatigue"));
	mp.insert(std::pair<int,std::string>(1,"non-fatigue"));

    //Read image folder
    std::string suffix = "jpg";
    std::vector<std::string> file_names;
    std::vector<std::string> path_list = readFileListSuffix(images_folder_path.c_str(), suffix.c_str(), file_names, false);
    int test_num = 0;
    for (int idx = 0; idx < path_list.size(); idx++) {
        //Read image
        std::string path_list_idx = path_list[idx];
        cv::Mat bgr_img = cv::imread(path_list_idx.c_str());
        //Model infer
        ClassInfo res = model.Infer(bgr_img);
		std::map<int,std::string>::iterator pos = mp.find(res.cls_idx);
		if(pos != mp.end()){
			std::cout << "The result of " << file_names[idx] << ": " << pos->second << "," << res.cls_score << std::endl;
		}else{
            std::cout << "Error!" << std::endl;
        }
	}
}
