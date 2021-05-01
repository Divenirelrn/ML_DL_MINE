import torch
import torch.nn as nn
from utee import misc
from collections import OrderedDict


__all__ = ["Resnet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

urls =  {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


def conv3x3(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BisicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BisicBlock, self).__init__()
        m = OrderedDict()
        m["conv1"] = conv3x3(in_planes, planes, stride=stride)
        m["bn1"] = nn.BatchNorm2d(planes)
        m["relu1"] = nn.ReLU(inplace=True)
        m["conv2"] = conv3x3(planes, planes)
        m["bn2"] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.group1(x)
        if self.downsample:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        m = OrderedDict()
        m["conv1"] = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        m["bn1"] = nn.BatchNorm2d(planes)
        m["relu1"] = nn.ReLU(inplace=True)
        m["conv2"] = conv3x3(planes, planes, stride=1)
        m["bn2"] = nn.BatchNorm2d(planes)
        m["relu2"] = nn.ReLU(inplace=True)
        m["conv3"] = nn.Conv2d(planes, 4 * planes, kernel_size=1, stride=1, padding=0, bias=0)
        m["bn3"] = nn.BatchNorm2d(4 * planes)
        self.group1 = nn.Sequential(m)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.group1(x)
        if self.downsample:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(self, block, layers):
        super(Resnet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        #self.detect = nn.Sequential(
        #    nn.Linear(512 * 7 * 7, 4096),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #    nn.Linear(4096, 30 * 7 * 7)
        #)
        self.detect = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 30, kernel_size=3, padding=1),
            nn.BatchNorm2d(30)
        )

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = list()
        layers.append(block(self.in_planes, planes, stride=stride, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.detect(out)
        #out = out.view(out.size(0), 512 * 7 * 7)
        #out = self.detect(out)
        out = torch.sigmoid(out)
        out = out.view(out.size(0), 30, 14, 14).permute(0, 2, 3, 1)
        #print("out:", out.shape)

        return out


def resnet18(pretrained=False, model_root=None, **kwargs):
    model = Resnet(BisicBlock, [2,2,2,2], **kwargs)
    if pretrained:
        misc.load_state_dict(model, urls["resnet18"], model_root)

    return model


def resnet34(pretrained=False, model_root=None, **kwargs):
    model = Resnet(BisicBlock, [3,4,6,3], **kwargs)
    if pretrained:
        misc.load_state_dict(model, urls["resnet34"], model_root)

    return model


def resnet50(pretrained=False, model_root=None, **kwargs):
    model = Resnet(BottleNeck, [3,4,6,3], **kwargs)
    if pretrained:
        misc.load_state_dict(model, urls["resnet50"], model_root)

    return model


def resnet101(pretrained=False, model_root=None, **kwargs):
    model = Resnet(BottleNeck, [3,4,23,3], **kwargs)
    if pretrained:
        misc.load_state_dict(model, urls["resnet101"], model_root)

    return model


def resnet152(pretrained=False, model_root=None, **kwargs):
    model = Resnet(BottleNeck, [3,8,36,3], **kwargs)
    if pretrained:
        misc.load_state_dict(model, urls["resnet152"], model_root)

    return model


if __name__ == "__main__":
    model = resnet50(pretrained=True)

    img = torch.randn(1, 3, 448, 448)
    out = model(img)
    print(out.shape)
