import torch
import torch.nn as nn
import torchvision.models as models



def ResNeXt50_32x4d(classes=1000):
    resnext50 = models.resnext50_32x4d(pretrained=True)
    inchannel = resnext50.fc.in_features
    # 最后加一个分类器
    resnext50.fc = nn.Sequential(nn.Linear(inchannel,classes))

    return resnext50

def ResNeXt101_32x8d(classes=1000):
    resnext101_32x8d = models.resnext101_32x8d(pretrained=True)
    inchannel = resnext101_32x8d.fc.in_features
    # 最后加一个分类器
    resnext101_32x8d.fc = nn.Sequential(nn.Linear(inchannel,classes))
        
    return resnext101_32x8d


