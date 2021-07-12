from model import xception

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#import visdom
import onnx
import tensorboardX

from torch.nn import DataParallel as DP

tbWriter = tensorboardX.SummaryWriter("runs/xception_3")

#viz = visdom.Visdom()

device = torch.device("cuda")
acc_glob = 0.0

#data_train = MNIST('./data/mnist',
#                   download=True,
#                   transform=transforms.Compose([
#                       transforms.Resize((32, 32)),
#                       transforms.ToTensor()]))
#data_test = MNIST('./data/mnist',
#                  train=False,
#                  download=True,
#                  transform=transforms.Compose([
#                      transforms.Resize((32, 32)),
#                      transforms.ToTensor()]))a
"""
data_train = datasets.CIFAR10('./data/cifar10', download=True,
                              transform=transforms.Compose([
                                  transforms.Resize((32,32)),
                                  #transforms.RandomCrop(size=32, padding=4),
                                  transforms.RandomHorizontalFlip(0.5),
                                  #transforms.ColorJitter(brightness=0.5),
                                  #transforms.RandomAffine(0, translate=(0.2, 0.2)),
                                  #transforms.RandomVerticalFlip(0.3),
                                  #transforms.RandomRotation(degrees=[0, 180]),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean = [0.49139968, 0.48215827, 0.44653124],
                                  #std = [0.24703233, 0.24348505, 0.26158768])]))
                                  std = [0.2023, 0.1994, 0.2010])]))
data_test = datasets.CIFAR10('./data/cifar10', download=True,
                              train = False, 
                              transform=transforms.Compose([
                                  transforms.Resize((32, 32)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean = [0.49139968, 0.48215827, 0.44653124],
                                  #std = [0.24703233, 0.24348505, 0.26158768])]))
                                  std = [0.2023, 0.1994, 0.2010])]))
"""

transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        #transforms.RandomAffine(0, translate=(0.125, 0.125)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4883, 0.4551, 0.4174],
        std = [0.2261, 0.2210, 0.2215])
    ])

transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4883, 0.4551, 0.4174],
        std = [0.2261, 0.2210, 0.2215])
    ])

data_train = datasets.ImageFolder("/data/xiaojun/data/kaggle_dog_cat/training_set", transform=transforms_train)
data_test = datasets.ImageFolder("/data/xiaojun/data/kaggle_dog_cat/test_set", transform=transforms_test)

data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=64, num_workers=8)

net = xception()
net.to(device)
device_ids = [0,1,2,3]
net = DP(net, device_ids = device_ids)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters(), lr=2e-3)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

"""
cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}
"""


def train(epoch):
    #global cur_batch_win
    print("learning rate of epoch{}:{}".format(epoch, optimizer.param_groups[0]['lr']))
    batch_num = len(data_train_loader)
    net.train()
    loss_list, batch_list = [], []
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)
        avg_loss += loss.detach().cpu().item()
        #loss_list.append(loss.detach().cpu().item())
        #batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))
            tbWriter.add_scalar("train loss", loss.detach().cpu().item(), epoch * batch_num + i)

        # Update Visualization
        #if viz.check_connection():
        #    cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
        #                             win=cur_batch_win, name='current_batch_loss',
        #                             update=(None if cur_batch_win is None else 'replace'),
        #                             opts=cur_batch_win_opts)

        loss.backward()
        optimizer.step()

    scheduler.step()
    return avg_loss / batch_num


def test(epoch):
    batch_num = len(data_test_loader)
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    img_num = 0
    for i, (images, labels) in enumerate(data_test_loader):
        images, labels = images.to(device), labels.to(device)
        img_num += len(images)
        with torch.no_grad():
            output = net(images)
            loss = criterion(output, labels)
            #avg_loss += loss.sum()
            avg_loss += loss.detach().cpu().item()
            if i % 10 == 0:
                tbWriter.add_scalar("validation loss", loss.detach().cpu().item(), epoch * batch_num + i)
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss1 = avg_loss / len(data_test)
    acc = float(total_correct) / len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss1, acc))
    tbWriter.add_scalar("accuracy", acc, epoch)

    return avg_loss / batch_num, acc

def train_and_test(epoch):
    global acc_glob
    train_loss = train(epoch)
    val_loss, acc = test(epoch)
    print("Epoch{},train_loss:{}, val_loss:{}".format(epoch, train_loss, val_loss))
    tbWriter.add_scalars("train_val loss:", {"train_loss": train_loss, 
                                             "val_loss": val_loss}, epoch)

    #dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True).to(device)
    #torch.onnx.export(net, dummy_input, "last.onnx")

    #onnx_model = onnx.load("last.onnx")
    #onnx.checker.check_model(onnx_model)
    torch.save(net, "last.pt")

    if acc > acc_glob:
        #torch.onnx.export(net, dummy_input, "best.onnx")

        #onnx_model = onnx.load("best.onnx")
        #onnx.checker.check_model(onnx_model)
        torch.save(net, "best.pt")
        acc_glob = acc


def main():
    for e in range(1, 101):
        train_and_test(e)


if __name__ == '__main__':
    main()
    tbWriter.close()
