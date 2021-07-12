import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

transforms_train = transforms.Compose([
        transforms.Resize((227, 227)),
        #transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        #transforms.Normalize(mean = [0.49139968, 0.48215827, 0.44653124],
        #std = [0.24703233, 0.24348505, 0.26158768])]))
        #std = [0.2023, 0.1994, 0.2010])
    ])

transforms_test = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        #transforms.Normalize(mean = [0.49139968, 0.48215827, 0.44653124],
        #std = [0.2023, 0.1994, 0.2010])
    ])

data_train = datasets.ImageFolder("/data/xiaojun/data/kaggle_dog_cat/training_set", transform=transforms_train)
data_test = datasets.ImageFolder("/data/xiaojun/data/kaggle_dog_cat/test_set", transform=transforms_test)

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

nimages = 0
mean = 0.
std = 0.
for batch, _ in data_train_loader:
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    mean += batch.mean(2).sum(0) 
    std += batch.std(2).sum(0)

# Final step
mean /= nimages
std /= nimages

print(mean)
print(std)
