#include "mobilenet.h"
#include "helper.h"
#include <iostream>
#include <string>
#include <map>
#include <typeinfo>

int main(int argc, char **argv){
    //Image filder & model dir
    std::string val_folder_path = argv[1];
    std::string model_dir          = argv[2];

	//Map of class index & name
    std::map<int, std::string> mp;
    mp.insert(std::pair<int,std::string>(0,"fatigue"));
    mp.insert(std::pair<int,std::string>(1,"nonfatigue"));

    //Model
    MobileNet model;
    std::cout << "Model has initialized!" << std::endl;
    model.Reset(model_dir);

    //Image folder verify
	if (check_folder(val_folder_path, false)) {
        std::cout << "validation_folder_path: " << val_folder_path << std::endl;
    } else {
        std::cout << "validation_folder_path folder does not exist." << std::endl;
        return 1;
    }
    //Read sundirectories
    std::vector<std::string> subdir_names;
    std::vector<std::string> subdir_list = readSubdirList(val_folder_path.c_str(), subdir_names);
    int correct = 0;
    int total = 0;
    for(int sidx = 0; sidx < subdir_list.size(); sidx++){
        std::string cls_folder_path = subdir_list[sidx];
        std::string cls_name = subdir_names[sidx];
        //Read image folder
        std::string suffix = "jpg";
        std::vector<std::string> file_names;
        std::vector<std::string> file_list = readFileListSuffix(cls_folder_path.c_str(), suffix.c_str(), file_names, false);
        std::cout << "path_list size:" << file_list.size() << std::endl;
        std::cout << "file_names size:" << file_names.size() << std::endl;
        total += file_list.size();
        for (int idx = 0; idx < file_list.size(); idx++) {
            //Read image
            std::string path_list_idx = file_list[idx];
            std::cout << path_list_idx << std::endl;
            cv::Mat bgr_img = cv::imread(path_list_idx.c_str());
            //Model infer
            ClassInfo res = model.Infer(bgr_img);
            std::cout << res.cls_idx << ", " << res.cls_score << std::endl;

			std::map<int,std::string>::iterator pos = mp.find(res.cls_idx);

            std::cout << "--------" << pos->second << ", ------------" << cls_name << "-----------" << std::endl;
            std::cout << "Is equal: " << ((pos->second).compare(cls_name) == 0) << ", " << typeid((pos->second).compare(cls_name)).name() << std::endl;
			if((pos->second).compare(cls_name) == 0){
                std::cout << "Equal" << std::endl;
				correct += 1;
			}
	    }
    }
	std::cout << "The accuracy is: " << static_cast<double>(correct) / total << std::endl;
}
