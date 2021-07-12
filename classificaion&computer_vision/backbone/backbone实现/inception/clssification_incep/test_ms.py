import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import os, cv2
import numpy as np
from tqdm import tqdm

tencrop = transforms.TenCrop(224)

normalize = transforms.Normalize(mean = [0.4883, 0.4551, 0.4174],
                                 std = [0.2261, 0.2210, 0.2215])
resize_crop = transforms.Resize((224, 224))
resize_crop_mirror = transforms.RandomHorizontalFlip(1.0)

def img_prep(img_path, size):
    img = cv2.imread(img_path)
    w, h = img.shape[1], img.shape[0]
    w_new = h_new = size
    if w < h:
        img = cv2.resize(img, (w_new, int(h * w_new / w)))
    else:# w/w_new >= h/h_new, 按照h比例缩放，w_current = w * (h_new/h)  
        img = cv2.resize(img, (int(w * h_new / h), h_new))
    w, h = img.shape[1], img.shape[0]

    if w >= h:
        #line
        img_1 = img[:, :h, :]
        gap = w - h_new 
        img_mid = img[:, int(gap/2):int(gap/2)+h_new , :]
        img_2 = img[:, int(gap):, :]
    else:
        #column
        img_1 = img[:w, :, :]
        gap = h - w_new 
        img_mid = img[int(gap/2):int(gap/2)+w_new, :, :]
        img_2 = img[int(gap):, :, :]

    #cv2.imwrite("./temp/img1" + str(size) + ".jpg", img_1)
    #cv2.imwrite("./temp/img2" + str(size) + ".jpg", img_2)
    #cv2.imwrite("./temp/img_mid" + str(size) + ".jpg", img_mid)

    crops = []
    for imge in [img_1, img_mid, img_2]:
        imge = imge[:, :, ::-1].transpose(2, 0, 1)
        imge = np.ascontiguousarray(imge)
        imge = torch.from_numpy(imge).to(device)
        imge = imge / 255.0
        imge = normalize(imge)

        tencrops0 = list(tencrop(imge))
        tencrops = []
        for crop in tencrops0:
            crop = crop.unsqueeze(0)
            tencrops.append(crop)

        elev_crop = resize_crop(imge)
        elev_crop_mir = resize_crop_mirror(elev_crop)
        elev_crop = elev_crop.unsqueeze(0)
        elev_crop_mir = elev_crop_mir.unsqueeze(0)
        
        tencrops.append(elev_crop)
        tencrops.append(elev_crop_mir)

        temp = torch.cat(tencrops, dim=0)
        
        crops.append(temp)

    
    return torch.cat(crops, dim=0)

device = torch.device("cuda")

cls_reflect = {"cats": 0, "dogs": 1}

def test(img_folder):
    net = torch.load("./best.pt")
    net.to(device)
    net.eval()
    total_correct = 0
    total_num = 0
    classes = os.listdir(img_folder)
    for cls in classes:
        cls_dir = os.path.join(img_folder, cls)
        images = os.listdir(cls_dir)
        total_num += len(images)
        labels = torch.tensor(cls_reflect[cls]).to(device)
        for img in tqdm(images):
            if not img.endswith(".jpg"):
                continue

            img_path = os.path.join(cls_dir, img)
            crops = []
            for size in [256, 288, 320, 352]:
                img_preped = img_prep(img_path, size)
                crops.append(img_preped)

            crops = torch.cat(crops, dim=0)
            crops = crops.to(device)
            with torch.no_grad():
                output = net(crops)
            
            #average
            output = output.mean(dim=0)
            pred = output.detach().max(0)[1]
            #voting
            #pred = output.detach().max(1)[1]
            #pred = (pred.sum() >= 72).int()

            total_correct += pred.eq(labels.view_as(pred)).sum()
            
    print("Test accury:", float(total_correct) / total_num)
    
transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4883, 0.4551, 0.4174],
                             std = [0.2261, 0.2210, 0.2215])
    ])

data_test = datasets.ImageFolder("/data/xiaojun/data/kaggle_dog_cat/test_set", transform=transforms_test)

data_test_loader = DataLoader(data_test, batch_size=256, num_workers=8)

def test0():
    net = torch.load("./best.pt")
    net.to(device)
    net.eval()
    total_correct = 0
    for i, (images, labels) in enumerate(data_test_loader):
        images, labels = images.to(device), labels.to(device)
        print("labels:", labels)
        with torch.no_grad():
            output = net(images)
            pred = output.detach().max(0)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

    print("Test accury:", float(total_correct) / len(data_test))


if __name__ == "__main__":
    test("/data/xiaojun/data/kaggle_dog_cat/test_set")
    #test0()

