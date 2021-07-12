import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

data_train = datasets.CIFAR10('./data/cifar10', download=True,
                              transform=transforms.Compose([
                                  transforms.Resize((32, 32)),
                                  transforms.ToTensor(),
                                  #ransforms.Normalize(mean = [0.49139968, 0.48215827, 0.44653124],
                                  #std = [0.2023, 0.1994, 0.2010])]))
                                  transforms.Normalize(mean = [0.49139968, 0.48215827, 0.44653124],
                                  std = [0.2023, 0.1994, 0.2010])]))

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)

data_test = datasets.CIFAR10('./data/cifar10', download=True,
                              train = False,
                              transform=transforms.Compose([
                                  transforms.Resize((32, 32)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.49139968, 0.48215827, 0.44653124), 
                                                       std=(0.2023, 0.1994, 0.2010))]))

data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

device = torch.device("cuda")

def test():
    net = torch.load("./best.pt")
    net.to(device)
    net.eval()
    total_correct = 0
    for i, (images, labels) in enumerate(data_test_loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            output = net(images)
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

    print("Test accury:", float(total_correct) / len(data_test))

    total_correct = 0
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            output = net(images)
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

    print("Train accury:", float(total_correct) / len(data_train))

if __name__ == "__main__":
    test()

