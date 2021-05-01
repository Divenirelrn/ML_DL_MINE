import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from net import vgg16, vgg16_bn
#from model import resnet50, resnet18
from model_bak import resnet18
from yoloLoss import yoloLoss
from dataset import yoloDataset
from eval_voc import map_compute
#from balancedDataParallel.data_parallel import BalancedDataParallel
from utee import misc

# from visualize import Visualizer
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from tensorboardX import SummaryWriter
writer = SummaryWriter("runs/exp_resnet18_fc_addconv_lr_scheduler_5folder")

use_gpu = torch.cuda.is_available()
print("use_gpu:", use_gpu)
device = torch.device("cuda:0" if use_gpu else "cpu")

file_root = '/home/workspace/vocall'
learning_rate = 0.0005
num_epochs = 60
batch_size = 64
use_resnet = False
#if use_resnet:
#    net = resnet50()
#else:
#    net = vgg16_bn()

# net.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             #nn.Linear(4096, 4096),
#             #nn.ReLU(True),
#             #nn.Dropout(),
#             nn.Linear(4096, 1470),
#         )
net = resnet18(pretrained=True)
#net.fc = nn.Linear(512,1470)
# initial Linear
# for m in net.modules():
#     if isinstance(m, nn.Linear):
#         m.weight.data.normal_(0, 0.01)
#         m.bias.data.zero_()
#print(net)
#net.load_state_dict(torch.load('yolo.pth'))
#print('load pre-trined model')
#if use_resnet:
#    resnet = models.resnet50(pretrained=True)
#    new_state_dict = resnet.state_dict()
#    dd = net.state_dict()
#    for k in new_state_dict.keys():
        #print(k)
#        if k in dd.keys() and not k.startswith('fc'):
            #print('yes')
#            dd[k] = new_state_dict[k]
#    net.load_state_dict(dd)
#    print("pre-trained model loaded")
#else:
#    vgg = models.vgg16_bn(pretrained=True)
#    new_state_dict = vgg.state_dict()
#    dd = net.state_dict()
#    for k in new_state_dict.keys():
#        print(k)
#        if k in dd.keys() and k.startswith('features'):
#            print('yes')
#            dd[k] = new_state_dict[k]
#    net.load_state_dict(dd)
if True:
    #net.load_state_dict(torch.load('best.pth'))
    pth_file = os.path.join(os.path.dirname(__file__), "best.pth")
    #misc.load_weights(net, pth_file)
    net.load_state_dict(torch.load(pth_file))
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

#gpu0_bs = 8
#acc_grad = 2

criterion = yoloLoss(7,2,5,0.5)
if use_gpu:
    #net = BalancedDataParallel(gpu0_bs // acc_grad, net, dim=0).to(device)
    #net = nn.DataParallel(net).to(device)
    net.to(device)

net.train()
# different learning rate
params=[]
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=1e-3)
# optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

# train_dataset = yoloDataset(root=file_root,list_file=['voc12_trainval.txt','voc07_trainval.txt'],train=True,transform = [transforms.ToTensor()] )
train_dataset = yoloDataset(root=file_root,list_file=['voc2012.txt','voc2007.txt'],train=True,transform = [transforms.ToTensor()] )
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0, drop_last=True)
# test_dataset = yoloDataset(root=file_root,list_file='voc07_test.txt',train=False,transform = [transforms.ToTensor()] )
test_dataset = yoloDataset(root=file_root,list_file='voc2007test.txt',train=False,transform = [transforms.ToTensor()] )
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0, drop_last=True)
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
logfile = open('log.txt', 'w')

num_iter = 0
#vis = Visualizer(env='xiong')
best_test_loss = np.inf
best_map = -1

for epoch in range(num_epochs):
    net.train()
    if epoch == 30:
        learning_rate = 0.0001
    #if epoch == 2:
    #     learning_rate = 0.0025
    #if epoch == 3:
    #     learning_rate = 0.0075
    #if epoch == 4:
    #     learning_rate = 0.01
    #if epoch == 75:
    #    learning_rate=0.001
    #if epoch == 105:
    #    learning_rate=0.0001
    #optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate*0.1,momentum=0.9,weight_decay=1e-4)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    
    total_loss = 0.
    
    for i,(images,target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images,target = images.to(device),target.to(device)
        
        pred = net(images)
        loss = criterion(pred,target)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
            %(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
            num_iter += 1
            #vis.plot_train_val(loss_train=total_loss/(i+1))
            writer.add_scalar("train_loss", loss.item(), epoch * len(train_loader) + i + 1)

    #validation
    validation_loss = 0.0
    net.eval()
    for i,(images,target) in enumerate(test_loader):
        images = Variable(images,volatile=True)
        target = Variable(target,volatile=True)
        if use_gpu:
            images,target = images.to(device),target.to(device)
        
        pred = net(images)
        loss = criterion(pred,target)
        #validation_loss += loss.item()
        #validation_loss /= len(test_loader.dataset)
        print("Epoch [{}], validation loss: {}".format(epoch, loss.item()))
        #vis.plot_train_val(loss_val=validation_loss)
        writer.add_scalar("validation_loss", loss.item(), epoch * len(test_loader) + i + 1)
    
    #if best_test_loss > validation_loss:
    #    best_test_loss = validation_loss
    #    print('get best test loss %.5f' % best_test_loss)
    #    torch.save(net.state_dict(),'best.pth')
    #logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')  
    #logfile.flush()      
    torch.save(net.state_dict(),'yolo.pth')
    
    map_ = map_compute()
    print("Epoch: {}, map: {}".format(epoch, map_))
    writer.add_scalar("validation_map", map_, epoch)

    if map_ > best_map:
        best_map = map_
        print("best map: {}".format(best_map))
        torch.save(net.state_dict(),'best.pth')
