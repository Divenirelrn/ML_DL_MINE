import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(pretrained=True, num_classes=2).to(device)
model.train()
# for k, v in model.state_dict().items():
#     print(k, v.shape)
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()
])

train_dataset = ImageFolder("../hymenoptera_data/train", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataset = ImageFolder("../hymenoptera_data/val", transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
# train_dataset = datasets.MNIST(r"D:\files\AI\Projects\my_pytorch_model\data", train=True, transform=transform)
# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# val_dataset = datasets.MNIST(r"D:\files\AI\Projects\my_pytorch_model\data", train=False, transform=transform)
# val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

accuracy = 0.0
for epoch in range(10):
    for data, target in train_dataloader:
        data, target = data.to(device), target.to(device)
        # target = target.float().unsqueeze(1)
        # target_copy = 1 - target
        # target = torch.cat((target_copy, target), dim=1)
        output = model(data)
        loss = loss_fn(output, target)
        print("epoch:[{}], train_loss: {}".format(epoch, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    acc = 0
    for img, label in val_dataloader:
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            res = model(img)
        res = torch.sigmoid(res)
        res = torch.max(res, dim=1)[1]
        acc += torch.eq(res, label).sum().item()

    acc = acc / len(val_dataloader.dataset)
    print("Epoch [{}], accuracy: {}%".format(epoch, acc * 100))

    if acc > accuracy:
        torch.save(model.state_dict(), "best.pth")
        accuracy = acc
        print("Best model saved")

    model.train()


