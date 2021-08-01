import os
All_list = './Data/all_list.txt'
Train_list = './Data/train_list.txt'
Valid_list = './Data/valid_list.txt'
Annotation_path = './Data/Annotations'
Images_path = './Data/Images/'
Test_output = './Test_output'
Processed_path = './Data/Processed'
Weights_file= './Output/save.ckpt-100000'
#Weights_file = "Alexnet_weight/Alexnet"

Classes=['bus', 'microbus', 'microvan', 'minivan', 'suv', 'car', 'truck']
Class_num = 8#因为有背景在 所以比Classes多一项
Image_w = 1600
Image_h = 1200
Roi_num = 600#防止候选框太多所做的限定,在我的项目里我取的是合格框数量（150）的4倍

Batch_size = 1
Max_iter =  100000
Summary_iter = 10
Save_iter = 500

Initial_learning_rate = 0.00001
Decay_rate = 0.8
Decay_iter = 900
Staircase = True
Roi_threshold = 0.7
NMS_threshold = 0.05
