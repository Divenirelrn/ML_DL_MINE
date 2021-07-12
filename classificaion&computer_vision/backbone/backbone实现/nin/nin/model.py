import torch
import torch.nn as nn

def nin_block(in_channels,out_channels,kernel_size,strides,padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,strides,padding),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU())
    return blk
"""
self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            #nn.Dropout(global_params.dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(global_params.dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(global_params.dropout_rate),
            nn.Linear(4096, global_params.num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
"""



class Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.mlpconv1 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        #self.dropout1 = nn.Dropout(0.5)

        self.mlpconv2 = nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        #self.dropout2 = nn.Dropout(0.5)

        self.mlpconv3 = nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True))
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.dropout3 = nn.Dropout(0.5)

        self.mlpconv4 = nn.Sequential(
                nn.Conv2d(384, num_classes, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes, num_classes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes, num_classes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True))
        #self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.dropout4 = nn.Dropout(0.5)

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=0)

        
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #        if m.bias is not None:
        #            nn.init.constant_(m.bias, 0)
            #elif isinstance(m, nn.BatchNorm2d):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)
            #elif isinstance(m, nn.Linear):
            #    nn.init.normal_(m.weight, 0, 0.01)
            #    nn.init.constant_(m.bias, 0)
        


    def forward(self, x):
        #print("x:", x[0, 0])
        out = self.mlpconv1(x)
        out = self.maxpool1(out)
        #out = self.dropout1(out)
        #print("layer1:", out)
        out = self.mlpconv2(out)
        out = self.maxpool2(out)
        #out = self.dropout2(out)
        #print("layer2:", out)
        out = self.mlpconv3(out)
        out = self.maxpool3(out)
        out = self.dropout3(out)
        #print("layer3:", out)
        #out = self.mlpconv4[0](out)
        #out = self.mlpconv4[1](out)
        #print("layer4(1):", out)
        #out = self.mlpconv4[2](out)
        #out = self.mlpconv4[3](out)
        #print("layer4(2):", out)
        #out = self.mlpconv4[4](out)
        out = self.mlpconv4(out)
        #print("layer4:", out)
        #out = self.dropout4(out)
        #out = self.mlpconv5(out)
        #out = self.maxpool5(out)
        #out = self.dropout5(out)
        #print("dropout5:", out.shape)
        #out = self.mlpconv6(out)
        out = self.avgpool(out)
        #print("out:", out)
        
        out = out.view(-1, 2)
        #print("out_view:", out)
        #out2 = out.view(-1, 2)
        #print("out2:", out2)

        return out

if __name__ == "__main__":
    model = Net()
    #print(dir(model))
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            for name2, param in module.named_parameters():
                print(name, param)

    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(out)
