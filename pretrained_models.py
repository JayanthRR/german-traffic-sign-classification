import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torchvision import models
from transforms_helper import *


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x
    

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def get_pretrained_inception(num_classes, pretrained=True):
    inception = torchvision.models.inception_v3(pretrained=pretrained)
    
    fc_in_features = inception.fc.in_features
    inception.fc = nn.Linear(in_features=fc_in_features, out_features=num_classes)
    inception.AuxLogits = InceptionAux(in_channels=768, num_classes=num_classes)
    
    return inception


def load_model(pretrain, feature_extractor=False):
    auxloss = False
    if pretrain == 'inception':

        model = get_pretrained_inception(num_classes=43, pretrained=True)
        if feature_extractor:
            for param in model.parameters():
                param.requires_grad = False

        auxloss = True
        train_transform = inception_train_transform
        val_transform = inception_test_transform
        test_transform = inception_test_transform

    elif pretrain == 'squeeze':
        model = models.squeezenet1_1(pretrained=True)
        if feature_extractor:
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.classifier._modules['1'].in_channels
        # model.classifier._modules['1'] = nn.Linear(num_ftrs, 42)
        model.classifier._modules['1'] = nn.Conv2d(num_ftrs, 43, 3)
        model.num_classes = 43
        train_transform = squeeze_train_transform
        val_transform = squeeze_test_transform
        test_transform = squeeze_test_transform

    elif pretrain == 'vgg':
        model = models.vgg11(pretrained=True)
        if feature_extractor:
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.classifier._modules['6'].in_features
        model.classifier._modules['6'] = nn.Linear(num_ftrs, 43)

        train_transform = resnet_train_transform
        val_transform = resnet_test_transform
        test_transform = resnet_test_transform

    elif pretrain == 'resnet34':
        model = models.resnet34(pretrained=True)
        if feature_extractor:
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 43)

        train_transform = resnet_train_transform
        val_transform = resnet_test_transform
        test_transform = resnet_test_transform

    else:
        model = models.resnet18(pretrained=True)
        if feature_extractor:
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 43)

        train_transform = resnet_train_transform
        val_transform = resnet_test_transform
        test_transform = resnet_test_transform

    return train_transform, val_transform, test_transform, auxloss
