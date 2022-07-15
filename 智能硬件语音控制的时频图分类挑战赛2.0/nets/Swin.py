import torch
import timm
from torchinfo import summary


model_swin_pretrained = timm.list_models('*swin*',pretrained=True)
# print(model_swin_pretrained)


def swin_tiny_patch4_window7_224(num_classes = 1000 , pretrained = True):
    model = timm.create_model('swin_tiny_patch4_window7_224',pretrained=pretrained,num_classes = num_classes)
    return model

def swin_small_patch4_window7_224(num_classes = 1000 , pretrained = True):
    model = timm.create_model('swin_small_patch4_window7_224',pretrained=pretrained,num_classes = num_classes)
    return model

def swin_base_patch4_window7_224(num_classes = 1000 , pretrained = True):
    model = timm.create_model('swin_base_patch4_window7_224_in22k',pretrained=pretrained,num_classes = num_classes)
    return model

def swin_base_patch4_window12_384(num_classes = 1000 , pretrained = True):
    model = timm.create_model('swin_base_patch4_window12_384_in22k',pretrained=pretrained,num_classes = num_classes)
    return model

def swin_large_patch4_window7_224(num_classes = 1000 , pretrained = True):
    model = timm.create_model('swin_large_patch4_window7_224_in22k',pretrained=pretrained,num_classes = num_classes)
    return model

def swin_large_patch4_window12_384(num_classes = 1000 , pretrained = True):
    model = timm.create_model('swin_large_patch4_window12_384_in22k',pretrained=pretrained,num_classes = num_classes)
    return model



def test():
    model = swin_large_patch4_window7_224(3)
    summary(model,(1,3,224,224))

# test()