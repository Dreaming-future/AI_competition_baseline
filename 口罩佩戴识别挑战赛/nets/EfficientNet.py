import torch
import timm
from torchinfo import summary


model_efficientnet = timm.list_models('*efficient*',pretrained=True)
print(model_efficientnet)
def EfficientNet_b0(num_classes = 1000, pretrained = True):
    model = timm.create_model('efficientnet_b0', num_classes = num_classes, pretrained = pretrained)
    return model

def EfficientNet_b1(num_classes = 1000, pretrained = True):
    model = timm.create_model('efficientnet_b1', num_classes = num_classes, pretrained = pretrained)
    return model
def EfficientNet_b2(num_classes = 1000, pretrained = True):
    model = timm.create_model('efficientnet_b2', num_classes = num_classes, pretrained = pretrained)
    return model
def EfficientNet_b3(num_classes = 1000, pretrained = True):
    model = timm.create_model('efficientnet_b3', num_classes = num_classes, pretrained = pretrained)
    return model

def EfficientNet_b4(num_classes = 1000, pretrained = True):
    model = timm.create_model('efficientnet_b4', num_classes = num_classes, pretrained = pretrained)
    return model

def EfficientNet_b5(num_classes = 1000, pretrained = True):
    model = timm.create_model('efficientnet_b5', num_classes = num_classes, pretrained = pretrained)
    return model
def EfficientNet_b6(num_classes = 1000, pretrained = True):
    model = timm.create_model('efficientnet_b6', num_classes = num_classes, pretrained = pretrained)
    return model
def EfficientNet_b7(num_classes = 1000, pretrained = True):
    model = timm.create_model('efficientnet_b7', num_classes = num_classes, pretrained = pretrained)
    return model
def EfficientNet_b8(num_classes = 1000, pretrained = True):
    model = timm.create_model('efficientnet_b8', num_classes = num_classes, pretrained = pretrained)
    return model

def test():   
    model = EfficientNet_b4()
    summary(model,(1,3,224,224))
test()