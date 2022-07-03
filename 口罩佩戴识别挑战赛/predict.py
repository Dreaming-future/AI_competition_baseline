
from nets.ConvNeXt import convnext_base
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import torch



class Prediction():
    def __init__(self):
        super(Prediction, self).__init__()
        image_mean = [0.38753143, 0.36847523, 0.27735737]
        image_std = [0.2023, 0.1994, 0.2010]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std)
        ])
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['mask_weared_incorrect', 'with_mask', 'without_mask']

    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

        net_type = 'ConvNeXt-XL'
        
        # from nets.DenseNet import DenseNet201,DenseNet169
        # self.net = DenseNet201(3)
        from nets.ConvNeXt import convnext_tiny,convnext_base,convnext_large,convnext_xlarge
        if net_type == 'ConvNeXt-T':
            self.net = convnext_tiny(3)
        elif net_type == 'ConvNeXt-B':
            self.net = convnext_base(3)
        elif net_type == 'ConvNeXt-L':
            self.net = convnext_large(3)
        elif net_type == 'ConvNeXt-XL':
            self.net = convnext_xlarge(3)
        checkpoint = torch.load('./checkpoint/best_{}_ckpt.pth'.format(net_type))
        self.net.load_state_dict(checkpoint['net'])
        print('训练时，最佳的准确率的结果为 {} %'.format(checkpoint['acc']))
        self.net.eval()

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"image_path": "./data/input/image/033.cd/033_0068.jpg""}
        :return: 模型预测成功后，直接返回预测的结果 {"label": 0}
        '''
        # return {"label": 0}
        img = Image.open(image_path).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)

        if self.use_gpu:
            self.net = self.net.cuda()
            out = self.net(img.unsqueeze(dim=0).cuda())
            pred_label = np.argmax(out.cpu().detach().numpy())
        else:
            out = self.net(img.unsqueeze(dim=0))
            pred_label = np.argmax(out.detach().numpy())

        # print("label:", pred_label)
        return {"label": self.classes[pred_label]}

import pandas as pd
test = pd.read_csv('./data/sample_submit.csv')

root = './data/test//'
num_classes = 3
imgs = os.listdir('./data/test/')

pred = Prediction()
pred.load_model()
pred.predict(image_path=root + imgs[1])
for i,img_path in enumerate(test['path']):
    # print(img_path)
    pre = pred.predict(image_path=root + img_path)
    test.iloc[i][1] = pre['label']

test.to_csv('submit.csv',index=False)