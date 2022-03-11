# Based on the ResNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import math
import torch
from torch import nn
from torchvision.models.resnet import conv3x3
from self_adapting_tensorization.adaptiveconv import SelfAdaptiveConv


class mySequential(nn.Sequential):
    def forward(self, *inputs, **kwargs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs, **kwargs)
            else:
                inputs = module(inputs, **kwargs)
        return inputs


class AdaptConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, order=2, stride=1, padding=1, bias=False):
        super(AdaptConv, self).__init__()
        self.conv1 = SelfAdaptiveConv(in_channels, out_channels, kernel_size=kernel_size, order=order, stride=stride,
                                      padding=padding, bias=bias)
        self.adaptive_weights_preconv = nn.Parameter(torch.ones(1, self.conv1.rank))
        self.adaptive_weights_preconv.data.normal_()

    def forward(self, x, adapt=False):
        if not adapt:
            l2_norm = torch.linalg.vector_norm(self.adaptive_weights_preconv)
            adaptive_weights = self.adaptive_weights_preconv.repeat(x.shape[0], 1)
            adaptive_weights += torch.randn_like(adaptive_weights) * l2_norm * 0.1
            x = self.conv1(x, adaptive_weights)
        else:
            adaptive_weights = self.adaptive_weights_preconv.repeat(x.shape[0], 1)
            x = self.conv1(x, adaptive_weights)
        return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride

        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = SelfAdaptiveConv.from_conv(conv3x3(inplanes, planes, stride))
        self.adaptive_weights_conv1 = nn.Parameter(torch.ones(1, self.conv1.rank))
        self.adaptive_weights_conv1.data.normal_()

        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = SelfAdaptiveConv(planes, planes, kernel_size=3, order=2, stride=1, padding=1, bias=False)
        self.adaptive_weights_conv2 = nn.Parameter(torch.ones(1, self.conv2.rank))
        self.adaptive_weights_conv2.data.normal_()

    def forward(self, x, adapt=False):
        batch_size = x.shape[0]
        residual = x
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        if not adapt:
            l2_norm = torch.linalg.vector_norm(self.adaptive_weights_conv1)
            adaptive_weights = self.adaptive_weights_conv1.repeat(batch_size, 1)
            adaptive_weights += torch.randn_like(adaptive_weights) * l2_norm * 0.1
            residual = self.conv1(residual, adaptive_weights)
        else:
            adaptive_weights = self.adaptive_weights_conv1.repeat(batch_size, 1)
            residual = self.conv1(residual, adaptive_weights)

        residual = self.bn2(residual)
        residual = self.relu2(residual)
        if not adapt:
            l2_norm = torch.linalg.vector_norm(self.adaptive_weights_conv2)
            adaptive_weights = self.adaptive_weights_conv2.repeat(batch_size, 1)
            adaptive_weights += torch.randn_like(adaptive_weights) * l2_norm * 0.1
            residual = self.conv2(residual, adaptive_weights)
        else:
            adaptive_weights = self.adaptive_weights_conv2.repeat(batch_size, 1)
            residual = self.conv2(residual, adaptive_weights)

        if self.downsample is not None:
            x = self.downsample(x)
        return x + residual


class Downsample(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(Downsample, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        assert nOut % nIn == 0
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)


class ResNetCifar(nn.Module):
    def __init__(self, depth, width=1, classes=10, channels=3, norm_layer=nn.BatchNorm2d):
        assert (depth - 2) % 6 == 0  # depth is 6N+2
        self.N = (depth - 2) // 6
        super(ResNetCifar, self).__init__()

        # Following the Wide ResNet convention, we fix the very first convolution
        self.conv1 = AdaptConv(channels, 16, kernel_size=3, order=2, stride=1, padding=1, bias=False)
        self.inplanes = 16
        self.layer1 = self._make_layer(norm_layer, 16 * width)
        self.layer2 = self._make_layer(norm_layer, 32 * width, stride=2)
        self.layer3 = self._make_layer(norm_layer, 64 * width, stride=2)
        self.bn = norm_layer(64 * width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * width, classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, norm_layer, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Downsample(self.inplanes, planes, stride)
        layers = [BasicBlock(self.inplanes, planes, norm_layer, stride, downsample)]
        self.inplanes = planes
        for i in range(self.N - 1):
            layers.append(BasicBlock(self.inplanes, planes, norm_layer))
        return mySequential(*layers)

    def forward(self, x, adapt=False):
        x = self.conv1(x, adapt=adapt)
        x = self.layer1(x, adapt=adapt)
        x = self.layer2(x, adapt=adapt)
        x = self.layer3(x, adapt=adapt)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
