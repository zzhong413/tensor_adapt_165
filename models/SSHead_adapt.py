from torch import nn
import math
import copy
from models.ResNet_tensor_adapt_init_modules import BasicBlock, AdaptConv


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


def head_on_layer2(net, width, classes):
    head = copy.deepcopy([net.layer3, net.bn, net.relu, net.avgpool])
    head.append(ViewFlatten())
    head.append(nn.Linear(64 * width, classes))
    return sshSequential(*head)
