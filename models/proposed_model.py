'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y


def make_layers(cfg, in_channels):
    layers = []
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


class proposed_net(nn.Module):
    def __init__(self, in_channels):
        super(proposed_net, self).__init__()
        self.features_branch_1a = make_layers([32, 32, 64, 'M'], in_channels)
        self.features_branch_2a = make_layers([16, 16, 24, 'M'], in_channels)
        self.features_branch_3a = make_layers([16, 16, 24, 'M'], in_channels)
        self.features_branch_4a = make_layers([16, 16, 16, 'M'], in_channels)
        self.features_branch_5a = make_layers([32, 32, 32, 'M'], in_channels)

        self.att_layer_1ax = PALayer(64)
        self.att_layer_2ax = PALayer(24)
        self.att_layer_3ax = PALayer(24)
        self.att_layer_4ax = PALayer(16)
        self.att_layer_5ax = PALayer(32)
        
        self.conv_out = make_layers([128, 128, 'M', 256, 256, 'M', 512, 512, 'M'], 160)
        self.classifier = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 87)
        )

    def forward(self, x1, x2, x3, x4, x5):
        features_1a = self.features_branch_1a(x1)
        features_2a = self.features_branch_2a(x2)
        features_3a = self.features_branch_3a(x3)
        features_4a = self.features_branch_4a(x4)
        features_5a = self.features_branch_5a(x5)
        att1a = self.att_layer_1ax(features_1a)
        att2a = self.att_layer_2ax(features_2a)
        att3a = self.att_layer_3ax(features_3a)
        att4a = self.att_layer_4ax(features_4a)
        att5a = self.att_layer_5ax(features_5a)

        g = torch.cat((att1a, att2a, att3a, att4a, att5a), dim=1)
        out = self.conv_out(g)
        out = self.classifier(out.view(out.size(0), -1))
        return out


if __name__ == '__main__':
    net = proposed_net(3)
    x = torch.randn(1,15,64,64)
    x1 = x[:, 0:3, :, :]
    x2 = x[:, 3:6, :, :]
    x3 = x[:, 6:9, :, :]
    x4 = x[:, 9:12, :, :]
    x5 = x[:, 12:15, :, :]
    
    print(net(Variable(x1), Variable(x2), Variable(x3), Variable(x4), Variable(x5)).size())

