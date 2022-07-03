import torch
import torch.nn as nn
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ResNeXt50(classes=1000):
    resnext50 = models.resnext50_32x4d(pretrained=True)
    for param in resnext50.parameters():
        param.requires_grad = False
        
    # 最后加一个分类器
    resnext50.fc = nn.Sequential(nn.Linear(2048,classes))
    for param in resnext50.fc.parameters():
        param.requires_grad = True
        
    resnext50 = resnext50.to(device)
    return resnext50

