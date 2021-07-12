import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import datasets, transforms

import os
import cv2
import numpy as np

from tqdm import tqdm

test_dir = "/data/xiaojun/data/kaggle_dog_cat/test_set/cats"
test_imgs = os.listdir(test_dir) 

device = torch.device("cuda")

transform = transforms.Compose([transforms.Normalize(mean = [0.4883, 0.4551, 0.4174],
                                 std = [0.2261, 0.2210, 0.2215])])

#cls_reflect = {0: 'cat', 1: 'dog'}

def image_prep(img):
    img = cv2.resize(img, (224, 224))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img / 255.0
    img = transform(img)

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img


def image_plot(img, score, cls, img_name):
    c1 = (0, 0)
    color = [100, 0, 0]
    label = cls_reflect[cls] + ":" + str(score)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    
    cv2.imwrite("./results/" + img_name, img)


def detect():
    net = torch.load("./best.pt")
    net.to(device)
    net.eval()

    for img in tqdm(test_imgs):
        if not img.endswith(".jpg"):
            continue

        img_path = os.path.join(test_dir, img)
        org_img = cv2.imread(img_path)

        prep_img = image_prep(org_img)

        with torch.no_grad():
            output = net(prep_img)
            output = F.softmax(output, dim=1)

        score, cls = round(output.max(dim=1)[0].item(), 4), output.max(dim=1)[1].item()
        print(cls)
        #image_plot(org_img, score, cls, img)


if __name__ == "__main__":
    detect()

