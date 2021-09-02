import torch
import torch.nn as nn
from extension.normalization import Norm

# vgg16, 可以看到, 带有参数的刚好为16个
net_arch16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M5', "FC1", "FC2", "FC"]

# vgg19, 基本和 vgg16 相同, 只不过在后3个卷积段中, 每个都多了一个卷积层
net_arch19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M5', "FC1", "FC2", "FC"]

def vgg(num_classes = 512, net_arch = net_arch16):
    return VGGNet(num_classes, net_arch)

class VGGNet(nn.Module):
    def __init__(self, num_classes, net_arch = net_arch16):
        # net_arch 即为上面定义的列表: net_arch16 或 net_arch19
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        layers = []
        in_channels = 1 # 初始化通道数
        for arch in net_arch:
            if arch == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif arch == 'M5':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            elif arch == "FC1":
                layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6))
                layers.append(nn.ReLU(inplace=True))
            elif arch == "FC2":
                layers.append(nn.Conv2d(1024,1024, kernel_size=1))
                layers.append(nn.ReLU(inplace=True))
            elif arch == "FC":
                layers.append(nn.Conv2d(1024,self.num_classes, kernel_size=1))
            elif arch == 'Norm':
                layers.append(Norm(in_channels))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=arch, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = arch
        self.vgg = nn.ModuleList(layers)

        
    def forward(self, input_data):
        x = input_data
        for layer in self.vgg:
            x = layer(x)
        out = x
        return out