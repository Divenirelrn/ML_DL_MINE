import torch
from torchvision import transforms
import cv2
from model import *
import glob
from utee import misc


model = resnet18(num_classes=2)
state_dict = torch.load("best.pth")
model.load_state_dict(torch.load("best.pth"))
model.eval()

img_dir = r"D:\files\AI\Projects\deeplearning_models\classification\hymenoptera_data\val\bees"

classes = {0: "ant", 1: "bee"}

transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor()
])

num = 0
for img_file in glob.glob(img_dir + "/*.jpg"):
    img = cv2.imread(img_file)
    img = cv2.resize(img, (224, 224))
    img = img[:,:,::-1].transpose(2,0,1).copy()
    img = torch.from_numpy(img).unsqueeze(0) / 255.0

    out = model(img)

    res = torch.sigmoid(out)
    res = torch.max(res, 1)
    label = classes[res[1].item()]
    score = res[0].item()
    print(label, score)
    if label == "bee":
        num += 1

print(num)