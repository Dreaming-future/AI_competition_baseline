import torch
import timm
from torchinfo import summary

model_bit_pretrained = timm.list_models('*bit*',pretrained=True)
# print(model_bit_pretrained)

def BiT_M_resnet152x4(num_classes = 1000, pretrained = True):
    model = timm.create_model('resnetv2_152x4_bitm_in21k', pretrained = pretrained, num_classes = num_classes)
    return model

def BiT_M_resnet152x2(num_classes = 1000, pretrained = True):
    model = timm.create_model('resnetv2_152x2_bitm_in21k', pretrained = pretrained, num_classes = num_classes)
    return model

def BiT_M_resnet101x3(num_classes = 1000, pretrained = True):
    model = timm.create_model('resnetv2_101x3_bitm_in21k', pretrained = pretrained, num_classes = num_classes)
    return model

def BiT_M_resnet101x1(num_classes = 1000, pretrained = True):
    model = timm.create_model('resnetv2_101x1_bitm_in21k', pretrained = pretrained, num_classes = num_classes)
    return model

def BiT_M_resnet50x1(num_classes = 1000, pretrained = True):
    model = timm.create_model('resnetv2_50x1_bitm_in21k', pretrained = pretrained, num_classes = num_classes)
    return model

def BiT_M_resnet50x3(num_classes = 1000, pretrained = True):
    model = timm.create_model('resnetv2_50x3_bitm_in21k', pretrained = pretrained, num_classes = num_classes)
    return model