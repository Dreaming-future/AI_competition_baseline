import timm
from torchinfo import summary


model_deit_pretrained = timm.list_models('*deit*',pretrained=True)
# print(model_deit_pretrained)

def DeiT_B(num_classes = 1000, pretrained = True):
    model = timm.create_model('deit_base_patch16_224',pretrained=pretrained,num_classes = num_classes)
    return model

def DeiT_S(num_classes = 1000, pretrained = True):
    model = timm.create_model('cdeit_small_patch16_224',pretrained=pretrained,num_classes = num_classes)
    return model

def DeiT_T(num_classes = 1000, pretrained = True):
    model = timm.create_model('deit_tiny_patch16_224',pretrained=pretrained,num_classes = num_classes)
    return model