import torch
import timm
from torchinfo import summary


model_vit_pretrained = timm.list_models('*vit*',pretrained=True)
print(model_vit_pretrained)

def Vit_bash_patch8_224(num_classes = 1000 , pretrained = True):
    model = timm.create_model('vit_base_patch8_224_in21k',pretrained=pretrained,num_classes = num_classes)
    return model

def Vit_bash_patch16_224(num_classes = 1000 , pretrained = True):
    model = timm.create_model('vit_base_patch16_224_miil_in21k',pretrained=pretrained, num_classes = num_classes)
    return model

def Vit_large_patch16_224(num_classes = 1000 , pretrained = True):
    model = timm.create_model('vit_large_patch16_224_in21k', pretrained=pretrained, num_classes = num_classes)
    return model

def Vit_large_patch32_224(num_classes = 1000 , pretrained = True):
    model = timm.create_model('vit_large_patch32_224_in21k', pretrained=pretrained, num_classes = num_classes)
    return model

def Vit_huge_patch14_224(num_classes = 1000 , pretrained = True):
    model = timm.create_model('vit_huge_patch14_224_in21k', pretrained=pretrained, num_classes = num_classes)
    return model

def test():
    model = Vit_bash_patch16_224(3)
    summary(model,(1,3,224,224))

# test()