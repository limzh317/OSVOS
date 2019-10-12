import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as torchmodel
import torch.nn as nn
from OSVOS.libs.vision import ResNet101, ResNet50, ResNet18, ResNet152, ResNet34
from torch.autograd import Variable
import ipdb


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.shortconv = nn.Conv2d(inplanes, planes*4, kernel_size=1, bias=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortconv(residual)
        out = self.relu(out)
        return out


class Decoder1(nn.Module):

    def __init__(self, is_init_decoder=True):
        super(Decoder1, self).__init__()
        self.is_init_decoder = True
        self.uplayer4 = Bottleneck(inplanes=2048 + 1024, planes=512, stride=1).cuda()
        self.uplayer3 = Bottleneck(inplanes=2048 + 512, planes=256, stride=1).cuda()
        self.uplayer2 = Bottleneck(inplanes=1024 + 256, planes=128, stride=1).cuda()
        self.uplayer1 = Bottleneck(inplanes=512 + 64, planes=64, stride=1).cuda()
        self.uplayer0 = Bottleneck(inplanes=256, planes=32, stride=1).cuda()
        self.outconv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, bias=True, stride=1)

        if self.is_init_decoder:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal(m.weight, mode='fan_in')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)

    def forward(self, input, residual):

        x5, x4, x3, x2, x1 = input[4], input[3], input[2], input[1], input[0]
        x5_up = F.upsample(x5, size=x4.size()[2:], mode='bilinear')
        x5_cat = torch.cat((x4, x5_up), dim=1)
        x5_out = self.uplayer4(x5_cat)

        x4_up = F.upsample(x5_out, size=x3.size()[2:], mode='bilinear')
        x4_cat = torch.cat((x3, x4_up), dim=1)
        x4_out = self.uplayer3(x4_cat)

        x3_up = F.upsample(x4_out, size=x2.size()[2:], mode='bilinear')
        x3_cat = torch.cat((x2, x3_up), dim=1)
        x3_out = self.uplayer2(x3_cat)

        x2_up = F.upsample(x3_out, size=x1.size()[2:], mode='bilinear')

        x2_cat = torch.cat((x1, x2_up), dim=1)
        x2_out = self.uplayer1(x2_cat)

        x1_up = F.upsample(x2_out, size=residual.size()[2:], mode='bilinear')
        x1_out = self.uplayer0(x1_up)
        out = self.outconv(x1_out)

        return out


class ResUNet1(nn.Module):

    """
    This version do not contain the shortcut convolution and middle level supervision.
    Just the pure version of encoder and decoder mode.

    The encoder module are initialized with resnet-101 parameters pretrained on ImageNet
    The decoder module are initialized with kaiming normal distribution.
    """
    def __init__(self, is_init_encoder=True):
        super(ResUNet1, self).__init__()

        self.is_init_encoder = is_init_encoder
        self.encoder = ResNet101()
        self.decoder = Decoder1().cuda()

        if self.is_init_encoder:
            self.init_encoder()

    def init_encoder(self):

        pretrained_state = torchmodel.resnet101(pretrained=True).state_dict()
        model_state = self.encoder.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            k in model_state and v.size() == model_state[k].size()}
        model_state.update(pretrained_state)
        self.encoder.load_state_dict(model_state)
        self.encoder = self.encoder.cuda()

    def forward(self, input):

        x5, x4, x3, x2, x1 = self.encoder(input)
        feat = [x1, x2, x3, x4, x5]
        out = self.decoder(feat, input)
        return out


class Decoder2(nn.Module):

    def __init__(self, is_init_decoder=True):
        super(Decoder2, self).__init__()
        n_class = 5
        self.is_init_decoder = is_init_decoder
        self.shortconv4 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1, bias=True)
        self.shortbn4 = nn.BatchNorm2d(num_features=1024)
        self.shortconv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, bias=True)
        self.shortbn3 = nn.BatchNorm2d(num_features=512)
        self.shortconv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True)
        self.shortbn2 = nn.BatchNorm2d(num_features=256)
        self.shortconv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=True)
        self.shortbn1 = nn.BatchNorm2d(num_features=64)

        self.uplayer4 = Bottleneck(inplanes=2048 + 1024, planes=512, stride=1)
        self.uplayer3 = Bottleneck(inplanes=2048 + 512, planes=256, stride=1)
        self.uplayer2 = Bottleneck(inplanes=1024 + 256, planes=128, stride=1)
        self.uplayer1 = Bottleneck(inplanes=512 + 64, planes=64, stride=1)
        self.uplayer0 = Bottleneck(inplanes=256, planes=32, stride=1)

        self.outconv0 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, bias=True, stride=1, padding=1)
        self.outconvs0 = nn.Conv2d(in_channels=128, out_channels=n_class, kernel_size=1, bias=True, stride=1)
        self.outconv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, bias=True, stride=1, padding=1)
        self.outconvs1 = nn.Conv2d(in_channels=128, out_channels=n_class, kernel_size=1, bias=True, stride=1)
        self.outconv2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, bias=True, stride=1, padding=1)
        self.outconvs2 = nn.Conv2d(in_channels=128, out_channels=n_class, kernel_size=1, bias=True, stride=1)
        self.outconv3 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=3, bias=True, stride=1, padding=1)
        self.outconvs3 = nn.Conv2d(in_channels=128, out_channels=n_class, kernel_size=1, bias=True, stride=1)
        self.outconv4 = nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=3, bias=True, stride=1, padding=1)
        self.outconvs4 = nn.Conv2d(in_channels=128, out_channels=n_class, kernel_size=1, bias=True, stride=1)

        ## init the decoder parameters
        if self.is_init_decoder:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal(m.weight, mode='fan_in')
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
            print('init the decoder parameters')

    def forward(self, input, residual):

        x5, x4, x3, x2, x1 = input[4], input[3], input[2], input[1], input[0]

        middle_output = []

        x5_up = F.upsample(x5, size=x4.size()[2:], mode='bilinear')
        x4_skip = self.shortbn4(self.shortconv4(x4))
        x5_cat = torch.cat((x4_skip, x5_up), dim=1)
        x5_out = self.uplayer4(x5_cat)
        x5_out1 = self.outconvs4(F.upsample(self.outconv4(x5_out), size=residual.size()[2:], mode='bilinear'))
        middle_output.append(x5_out1)

        x4_up = F.upsample(x5_out, size=x3.size()[2:], mode='bilinear')
        x3_skip = self.shortbn3(self.shortconv3(x3))
        x4_cat = torch.cat((x3_skip, x4_up), dim=1)
        x4_out = self.uplayer3(x4_cat)
        x4_out1 = self.outconvs3(F.upsample(self.outconv3(x4_out), size=residual.size()[2:], mode='bilinear'))
        middle_output.append(x4_out1)

        x3_up = F.upsample(x4_out, size=x2.size()[2:], mode='bilinear')
        x2_skip = self.shortbn2(self.shortconv2(x2))
        x3_cat = torch.cat((x2_skip, x3_up), dim=1)
        x3_out = self.uplayer2(x3_cat)
        x3_out1 = self.outconvs2(F.upsample(self.outconv2(x3_out), size=residual.size()[2:], mode='bilinear'))
        middle_output.append(x3_out1)

        x2_up = F.upsample(x3_out, size=x1.size()[2:], mode='bilinear')
        x1_skip = self.shortbn1(self.shortconv1(x1))
        x2_cat = torch.cat((x1_skip, x2_up), dim=1)
        x2_out = self.uplayer1(x2_cat)
        x2_out1 = self.outconvs1(F.upsample(self.outconv1(x2_out), size=residual.size()[2:], mode='bilinear'))
        middle_output.append(x2_out1)

        x1_up = F.upsample(x2_out, size=residual.size()[2:], mode='bilinear')
        x1_out = self.uplayer0(x1_up)
        out = self.outconvs0(self.outconv0(x1_out))

        middle_output.append(out)

        return middle_output

class ResUNet2(nn.Module):

    """
    This version have slight defference with ResUNet2.
    shortcut convolution and bn
    middel level supervision

    """

    def __init__(self, is_init_encoder=True):
        super(ResUNet2, self).__init__()
        self.model = 'resnet101'

        self.is_init_encoder = is_init_encoder
        if self.model == 'resnet18':
            self.encoder = ResNet18()
        elif self.model == 'resnet34':
            self.encoder = ResNet34()
        elif self.model == 'resnet50':
            self.encoder = ResNet50()
        elif self.model == 'resnet101':
            self.encoder = ResNet101()
        elif self.model == 'resnet152':
            self.encoder = ResNet152()
        self.decoder = Decoder2().cuda()

        if self.is_init_encoder:
            self.init_encoder()
            print('init the encoder parameters')

    def init_encoder(self):
        if self.model == 'resnet18':
            pretrained_state = torchmodel.resnet18(pretrained=True).state_dict()
        elif self.model == 'resnet34':
            pretrained_state = torchmodel.resnet34(pretrained=True).state_dict()
        elif self.model == 'resnet50':
            pretrained_state = torchmodel.resnet50(pretrained=True).state_dict()
        elif self.model == 'resnet101':
            pretrained_state = torchmodel.resnet101(pretrained=True).state_dict()
        elif self.model == 'resnet152':
            pretrained_state = torchmodel.resnet152(pretrained=True).state_dict()

        model_state = self.encoder.state_dict()

        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            k in model_state and v.size() == model_state[k].size()}
        model_state.update(pretrained_state)
        self.encoder.load_state_dict(model_state)
        self.encoder = self.encoder.cuda()

    def forward(self, input):
        #ipdb.set_trace()
        residual = input
        x5, x4, x3, x2, x1 = self.encoder(input)
        feat = [x1, x2, x3, x4, x5]
        output = self.decoder(feat, residual)

        return output


if __name__ == '__main__':

    unet = ResUNet2().cuda()
    x = Variable(torch.FloatTensor(np.ones((1, 3, 400, 400))), volatile=True).cuda()
    x = unet(x)
    print(x.shape)






