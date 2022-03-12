from torch import nn
import torch
import math
import copy
from models.ResNet_tensor_adapt_TTT import BasicBlock, AdaptConv, Downsample, mySequential


class sshSequential(nn.Sequential):
    def forward(self, *inputs, **kwargs):
        for module in self._modules.values():
            # only adapt conv layers, other layers do not have adapt argument
            if isinstance(module, BasicBlock) or isinstance(module, AdaptConv):
                if type(inputs) == tuple:
                    inputs = module(*inputs, **kwargs)
                else:
                    inputs = module(inputs, **kwargs)
            else:
                if type(inputs) == tuple:
                    inputs = module(*inputs)
                else:
                    inputs = module(inputs)
        return inputs


class ViewFlatten(nn.Module):
    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ExtractorHead(nn.Module):
    def __init__(self, ext, head):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head

    def forward(self, x, adapt=False):
        if isinstance(self.head, nn.Linear):
            return self.head(self.ext(x, adapt=adapt))
        else:
            return self.head(self.ext(x, adapt=adapt), adapt=adapt)


def extractor_from_layer3(net):
    layers = [net.conv1, net.layer1, net.layer2, net.layer3,
              net.bn, net.relu, net.avgpool, ViewFlatten()]
    return sshSequential(*layers)


def extractor_from_layer2(net):
    layers = [net.conv1, net.layer1, net.layer2]
    return sshSequential(*layers)


# def head_on_layer2(net, width, classes):
#     head = copy.deepcopy([net.layer3, net.bn, net.relu, net.avgpool])
#     head.append(ViewFlatten())
#     head.append(nn.Linear(64 * width, classes))
#     return sshSequential(*head)


class Head_on_layer2(nn.Module):
    def __init__(self, width, classes, depth, norm_layer=nn.BatchNorm2d):
        checkpoint_dir = 'results/cifar10_layer2_gn_expand/ckpt.pth'
        self.ttt_net = torch.load(checkpoint_dir)['head']

        assert (depth - 2) % 6 == 0  # depth is 6N+2
        self.N = (depth - 2) // 6
        self.inplanes = 32
        super(Head_on_layer2, self).__init__()

        self.layer3 = self._make_layer(norm_layer, 64 * width, j='0.', stride=2)
        self.bn = norm_layer(64 * width)
        self.bn.weight.data = self.ttt_net['1.weight']
        self.bn.bias.data = self.ttt_net['1.bias']
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.vf = ViewFlatten()
        self.fc = nn.Linear(64 * width, classes)
        self.fc.weight.data = self.ttt_net['5.weight']
        self.fc.bias.data = self.ttt_net['5.bias']

    def _make_layer(self, norm_layer, planes, j, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Downsample(self.inplanes, planes, stride)
        layers = [BasicBlock(self.inplanes, planes, norm_layer, j=j, i=0, ttt_net=self.ttt_net, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for i in range(self.N - 1):
            layers.append(BasicBlock(self.inplanes, planes, norm_layer, j=j, i=i+1, ttt_net=self.ttt_net))
        return mySequential(*layers)

    def forward(self, x, adapt=False):
        x = self.layer3(x, adapt=adapt)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.vf(x)
        x = self.fc(x)
        return x
