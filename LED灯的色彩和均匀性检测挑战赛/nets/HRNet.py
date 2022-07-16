import timm
from torchinfo import summary

model_hrnet_pretrained = timm.list_models('*hr*',pretrained=True)
# print(model_cait_pretrained)

def hrnet_w18(num_classes = 1000, pretrained = True):
    model = timm.create_model('hrnet_w18',pretrained=pretrained,num_classes = num_classes)
    return model

def hrnet_w18_small(num_classes = 1000, pretrained = True):
    model = timm.create_model('hrnet_w18_small',pretrained=pretrained,num_classes = num_classes)
    return model

def hrnet_w18_small_v2(num_classes = 1000, pretrained = True):
    model = timm.create_model('hrnet_w18_small_v2',pretrained=pretrained,num_classes = num_classes)
    return model


def hrnet_w18_small_v2(num_classes = 1000, pretrained = True):
    model = timm.create_model('hrnet_w18_small_v2',pretrained=pretrained,num_classes = num_classes)
    return model

def hrnet_w30(num_classes = 1000, pretrained = True):
    model = timm.create_model('hrnet_w30',pretrained=pretrained,num_classes = num_classes)
    return model

def hrnet_w32(num_classes = 1000, pretrained = True):
    model = timm.create_model('hrnet_w32',pretrained=pretrained,num_classes = num_classes)
    return model

def hrnet_w40(num_classes = 1000, pretrained = True):
    model = timm.create_model('hrnet_w40',pretrained=pretrained,num_classes = num_classes)
    return model

def hrnet_w48(num_classes = 1000, pretrained = True):
    model = timm.create_model('hrnet_w48',pretrained=pretrained,num_classes = num_classes)
    return model

def hrnet_w64(num_classes = 1000, pretrained = True):
    model = timm.create_model('hrnet_w64',pretrained=pretrained,num_classes = num_classes)
    return model