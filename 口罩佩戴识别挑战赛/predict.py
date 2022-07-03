import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import torch
import torch.nn.functional as F

class Prediction():
    def __init__(self):
        super(Prediction, self).__init__()
        # 利用ImageNet的标准差和方差
        image_mean = [0.38753143, 0.36847523, 0.27735737]
        image_std = [0.2023, 0.1994, 0.2010]
        # 不进行数据增强的transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std)
        ])
        
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 3
        self.classes = ['mask_weared_incorrect', 'with_mask', 'without_mask']

    def load_model(self, net = 'ConvNeXt-B'):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        
        from nets.ConvNeXt import convnext_tiny,convnext_base,convnext_large,convnext_xlarge
        from nets.DenseNet import DenseNet201,DenseNet169

        print("我们使用的是 {} 模型进行测试".format(net))
        if net == 'ConvNeXt-T':
            self.net = convnext_tiny(self.num_classes)
        elif net == 'ConvNeXt-B':
            self.net = convnext_base(self.num_classes)
        elif net == 'ConvNeXt-L':
            self.net = convnext_large(self.num_classes)
        elif net == 'ConvNeXt-XL':
            self.net = convnext_xlarge(self.num_classes)
        elif net == 'DenseNet169':
            self.net = DenseNet169(self.num_classes)
        elif net == 'DenseNet201':
            self.net = DenseNet201(self.num_classes)
        
            
        checkpoint = torch.load('./checkpoint/best_{}_ckpt.pth'.format(net))
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
            # print(F.softmax(out)) # 得到概率
        else:
            out = self.net(img.unsqueeze(dim=0))
            pred_label = np.argmax(out.detach().numpy())

        # print("label:", pred_label)
        return {"label": self.classes[pred_label]}


def save_csv(path = 'submit.csv',net = 'ConvNeXt-B'):
    import pandas as pd
    test = pd.read_csv('./data/sample_submit.csv')

    pred = Prediction()
    pred.load_model(net)
    print("==> 开始测试文件中的图片,一共有{}张图片需要测试".format(len(test)))
    for i,img_path in enumerate(test['path']):
        pre = pred.predict(image_path=root + img_path)
        test.iloc[i][1] = pre['label'] # 写入标签
        if (i+1)%100 == 0:
            print("已测试完毕{}张图片".format(i+1))
    print("==> 测试完毕，正在保存文件 {}".format(path))
    test.to_csv( path, index=False)
    
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    root = './data/test//' # 文件夹的路径
    num_classes = 3 # 类别

    net = 'ConvNeXt-L'
    save_csv(path = 'submit_{}.csv'.format(net),net = net)