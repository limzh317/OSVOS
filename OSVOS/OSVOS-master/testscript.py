import numpy as np
import torchvision
import torch
import torch.nn as nn


def init_weights(net):
    for m in net.modules():
        if type(m) == nn.Conv2d:
            m.weight.data.normal_(0, 0.01)
            m.bias.data.fill_(0)
        elif type(m) == nn.BatchNorm2d:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def init_network():

    model = torchvision.models.resnet18(pretrained=False)
    init_weights(net=model)

    print (model)


if __name__ == '__main__':
    init_network()
