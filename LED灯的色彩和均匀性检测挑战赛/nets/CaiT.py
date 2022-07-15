import timm
from torchinfo import summary


model_cait_pretrained = timm.list_models('*cait*',pretrained=True)
# print(model_cait_pretrained)

def CaiT_s24(num_classes = 1000, pretrained = True):
    model = timm.create_model('cait_s24_224',pretrained=pretrained,num_classes = num_classes)
    return model

def CaiT_xxs24(num_classes = 1000, pretrained = True):
    model = timm.create_model('cait_xxs24_224',pretrained=pretrained,num_classes = num_classes)
    return model

def CaiT_xxs36(num_classes = 1000, pretrained = True):
    model = timm.create_model('cait_xxs36_224',pretrained=pretrained,num_classes = num_classes)
    return model