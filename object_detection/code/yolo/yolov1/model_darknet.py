import torch
import torch.nn as nn
from utee import misc
from collections import OrderedDict
from darknet import darknet_19

__all__ = ["Darknet"]

class Darknet(nn.Module):
    def __init__(self):
        super(Darknet, self).__init__()
        self.feature = darknet_19()

        self.detect = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 30 * 7 * 7)
        )

    def forward(self, x):
        out = self.feature(x)
        #print("backbone:", out.shape)
        out = out.view(out.size(0), 1024 * 7 * 7)
        out = self.detect(out)
        out = torch.sigmoid(out)
        out = out.view(out.size(0), 30, 7, 7).permute(0, 2, 3, 1)
        #print("out:", out.shape)

        return out


if __name__ == "__main__":
    model = resnet18(pretrained=True)

    img = torch.randn(1, 3, 224, 224)
    out = model(img)
    print(out.shape)
