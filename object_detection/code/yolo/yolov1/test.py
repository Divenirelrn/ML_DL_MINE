import torch
from torchvision import transforms,models
from torch.utils.data import DataLoader
from resnet_yolo import resnet50
from dataset import yoloDataset

#超参数
file_root = "../../data/vocall"
batch_size = 2
use_gpu = True

#载入模型
print("Loading model...")
net = resnet50(pretrained=False)
print(net)
#print(torch.load("./best.pth"))
#net.load_state_dict(torch.load("./best.pth"))
#new_state_dict = torch.load("./best.pth")

#state_dict =checkpoint['state_dict']
state_dict = torch.load("./best.pth")
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    print("k, v", k)
    if 'module' not in k:
        k = 'module.'+k
    else:
        k = k.replace('module.', '')
    new_state_dict[k]=v
net.load_state_dict(new_state_dict)

if use_gpu:
    net.cuda()
net.eval()
print("Model has loaded")

#载入数据
test_dataset = yoloDataset(root=file_root,list_file='voc2007test.txt',train=False,transform = [transforms.ToTensor()] )
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

for i, (images, target) in enumerate(test_loader):
    if use_gpu:
        images, target = images.cuda(), target.cuda()

    pred = net(images)
    #print(pred)
    print(pred[0, :, :, :])

    
    


