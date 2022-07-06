import torch
import timm
from torchinfo import summary


model_efficientnetv2 = timm.list_models('*efficientnetv2*',pretrained=True)

def Efficientv2_XL(num_classes = 1000, pretrained = True):
    model = timm.create_model('tf_efficientnetv2_xl_in21ft1k',pretrained=pretrained,num_classes = num_classes)
    return model
def Efficientv2_L(num_classes = 1000, pretrained = True):
    model = timm.create_model('tf_efficientnetv2_l_in21ft1k',pretrained=pretrained,num_classes = num_classes)
    return model
def Efficientv2_M(num_classes = 1000, pretrained = True):
    model = timm.create_model('tf_efficientnetv2_m_in21ft1k',pretrained=pretrained,num_classes = num_classes)
    return model
def Efficientv2_S(num_classes = 1000, pretrained = True):
    model = timm.create_model('tf_efficientnetv2_s_in21ft1k',pretrained=pretrained,num_classes = num_classes)
    return model

    
# def test():   
#     model = EfficientNet_b4()
#     summary(model,(1,3,224,224))
# test()