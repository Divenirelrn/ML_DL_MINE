import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        #transforms.Normalize(mean = [0.49139968, 0.48215827, 0.44653124],
        #std = [0.24703233, 0.24348505, 0.26158768])]))
        #std = [0.2023, 0.1994, 0.2010])
        transforms.Normalize(mean = [0.4883, 0.4551, 0.4174],
                             std = [0.2261, 0.2210, 0.2215])
    ])

transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean = [0.49139968, 0.48215827, 0.44653124],
        #std = [0.2023, 0.1994, 0.2010])
        transforms.Normalize(mean = [0.4883, 0.4551, 0.4174],
                             std = [0.2261, 0.2210, 0.2215])
    ])

data_train = datasets.ImageFolder("/data/xiaojun/data/kaggle_dog_cat/training_set", transform=transforms_train)
data_test = datasets.ImageFolder("/data/xiaojun/data/kaggle_dog_cat/test_set", transform=transforms_test)

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=256, num_workers=8)

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

