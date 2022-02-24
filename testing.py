from models.ResNet_tensor_adapt_init_modules import ResNetCifar as ResNet
import torch.nn as nn

net = ResNet(26, 1, channels=3, classes=10, norm_layer=nn.BatchNorm2d)

for parameter in net.parameters():
    print(parameter)