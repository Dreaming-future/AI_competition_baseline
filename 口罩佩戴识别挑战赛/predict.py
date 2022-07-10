import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from utils import Get_model

class Prediction():
    def __init__(self):
        super(Prediction, self).__init__()
        # 利用数据集的标准差和方差
        image_mean = [0.4940, 0.4187, 0.3855]
        image_std = [0.2048, 0.1941, 0.1932]
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
        self.Ensemble = False
        self.nets = [] # 集成模型的列表

    def load_model(self, net = 'ConvNeXt-B', type = 'best', threshold = 98.5):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

        print("我们使用的是 {} 模型进行测试".format(net))
        self.net = Get_model(net, num_classes=self.num_classes)

        checkpoint = torch.load(r'./checkpoint/{}_ckpt.pth'.format(net))
        acc = checkpoint['acc']
        print('训练时，一共迭代了{}次，最后一次的准确率大概是 {} %'.format(checkpoint['epoch'],acc))

        checkpoint_best = torch.load('./checkpoint/best_{}_ckpt.pth'.format(net))
        best_acc = checkpoint_best['acc']
        from utils import remove_prefix
        print('训练时，最佳的准确率的结果为 {} %'.format(best_acc))
        flag = True

        if type == 'best' and best_acc >= threshold:
            self.net.load_state_dict(remove_prefix(checkpoint_best['net'], 'module.'))
        elif type == 'last' and acc >= threshold:
            self.net.load_state_dict(remove_prefix(checkpoint['net'], 'module.'))
        else:
            self.net.load_state_dict(remove_prefix(checkpoint_best['net'], 'module.'))
            flag = False

        if self.use_gpu:
            self.net = torch.nn.DataParallel(self.net)
            self.net = self.net.cuda()
        self.net.eval()
        return flag

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
            out = self.net(img.unsqueeze(dim=0).cuda())
            pred_label = np.argmax(out.cpu().detach().numpy())
            prob = F.softmax(out,dim=1).cpu().detach().numpy()
        else:
            out = self.net(img.unsqueeze(dim=0))
            pred_label = np.argmax(out.detach().numpy())
            prob = F.softmax(out,dim=1).detach().numpy()

        # print("label:", pred_label)
        return {"label": pred_label, 'class':self.classes[pred_label],"label_prob":prob[0]}


def save_csv(path = 'submit.csv',net = 'ConvNeXt-B', type = 'vote'):
    import pandas as pd
    test = pd.read_csv('./data/sample_submit.csv')
    pred = Prediction()

    total = len(test)
    print("==> 开始测试文件中的图片,一共有{}张图片需要测试".format(total))
    if net == 'ensemble':
        # 一个一个的载入模型进行测试，测试成功后对投票结果进行整合
        import glob
        paths = glob.glob(r'./checkpoint/best*')
        NET = []
        if type == 'vote':
            test_vote = test.copy()
            for i in range(num_classes):
                test_vote['label_%d'%i] = 0
            for path in paths:
                net = path.split('_')[1]
                NET.append(net)
                # 判断载入的模型是否符合标准，如果达到了标准就进行测试
                flag = pred.load_model(net)
                if flag:
                    print("==> #--------------------------------------#")
                    print("==> # 筛选{}模型进入集成模型".format(net))
                    print("==> #--------------------------------------#")
                    with tqdm(total=total,desc=f'{net} Predict Pictures {total}',mininterval=0.3) as pbar:
                        for i,img_path in enumerate(test['path']):
                            pre = pred.predict(image_path=root + img_path)
                            test_vote.iloc[i,2 + pre['label']] += 1
                            pbar.update(1)
                    print(test_vote.head(5))
                else:
                    print("==> #--------------------------------------#")
                    print("==> # 【未筛选{}模型进入集成模型】".format(net))
                    print("==> #--------------------------------------#")
            # 对投票结果进行处理，取出最佳的投票结果
            test_vote = np.array(test_vote.iloc[:,2:])
            with tqdm(total=total,desc=f'最后集成处理》》》',mininterval=0.3) as pbar:
                for i,img_path in enumerate(test['path']):
                    c = np.argmax(test_vote[i])
                    test.iloc[i,1] = pred.classes[c]
        elif type == 'mean':
            count = 0
            test_mean = test.copy()
            for i in range(num_classes):
                test_mean['label_prob%d'%i] = 0
            for path in paths:
                net = path.split('_')[1]
                NET.append(net)
                # 判断载入的模型是否符合标准，如果达到了标准就进行测试
                flag = pred.load_model(net)
                if flag:
                    print("==> #--------------------------------------#")
                    print("==> # 筛选{}模型进入集成模型".format(net))
                    print("==> #--------------------------------------#")
                    with tqdm(total=total,desc=f'{net} Predict Pictures {total}',mininterval=0.3) as pbar:
                        for i,img_path in enumerate(test['path']):
                            pre = pred.predict(image_path=root + img_path)
                            for j,prob in enumerate(pre['label_prob']):
                                test_mean.iloc[i,2 + j] += prob
                            pbar.update(1)
                    print(test_mean.head(5))
                    count += 1
                else:
                    print("==> #--------------------------------------#")
                    print("==> # 【未筛选{}模型进入集成模型】".format(net))
                    print("==> #--------------------------------------#")
            # 对均值结果进行处理，取出最优结果
            test_mean = np.array(test_mean.iloc[:,2:])
            test_mean /= count
            for i,img_path in enumerate(test['path']):
                # 第一次后处理未涉及的难样本 index
                # 第一次后处理 - 将预测概率值大于 0.5 的样本作为分类的类别
                threshold = 0.5
                flag = False
                with tqdm(total=total,desc=f'最后集成处理》》》',mininterval=0.3) as pbar:
                    for index,prob in enumerate(test_mean):
                        if prob > threshold:
                            res = index
                            flag = True
                            break
                    # 进行第二次处理
                    if not flag:
                        res = np.argmax(test_mean[i])
                    test.iloc[i,1] = pred.classes[res]
            print("集成一共有个 {} 模型， 分别是 {}".format(len(NET), " ".join(NET)))
    else:
        pred.load_model(net)
        with tqdm(total=total,desc=f'Predict Pictures {total}',mininterval=0.3) as pbar:
            for i,img_path in enumerate(test['path']):
                pre = pred.predict(image_path=root + img_path)
                test.iloc[i,1] = pre['class']
                pbar.update(1)
            
    print("==> 测试完毕，正在保存文件 {}".format(path))
    test.to_csv( path, index=False)
    
import argparse    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification Predict')
    parser.add_argument('--net', '--model', '-net', '-model', type = str, choices=['LeNet5', 'AlexNet', 'VGG16','VGG19',
                                                       'ResNet34','ResNet50','ResNet101',   
                                                       'DenseNet','DenseNet121','DenseNet169','DenseNet201',
                                                       'MobileNetv1','MobileNetv2',
                                                       'ResNeXt50-32x4d','ResNeXt101-32x8d',
                                                       'EfficientNet-b0','EfficientNet-b1','EfficientNet-b2','EfficientNet-b3','EfficientNet-b4','EfficienNet-b5','EfficientNet-b6','EfficientNet-b7','EfficientNet-b8',
                                                       'EfficientNetv2-S','EfficientNetv2-M','EfficientNetv2-L','EfficientNetv2-XL',
                                                       'ConvNeXt-T','ConvNeXt-S','ConvNeXt-B','ConvNeXt-L','ConvNeXt-XL',
                                                       'Swin-M','Swin-L',
                                                       'ViT-B','ViT-L','ViT-H',
                                                       'CaiT-s24','CaiT-xxs24','CaiT-xxs36',
                                                       'DeiT-B','DeiT-T','DeiT-S',
                                                       'BiT-M-resnet152x4','BiT-M-resnet152x2','BiT-M-resnet101x3','BiT-M-resnet101x1',
                                                       'ensemble'], default='ensemble', help='net/model type')
    parser.add_argument('--type','-t',type=str, choices=['mean','vote'],default = 'vote',  help='Ensemble type')
    parser.add_argument('--num-classes', '-nc',type = int, default=3)
    args = parser.parse_args()
    print(args)

    root = r'./data/test//' # 文件夹的路径
    num_classes = args.num_classes # 类别
    net = args.net
    type = args.type
    from datetime import datetime
    save_csv(path = 'submit_{}_{}_{}.csv'.format(net,type,datetime.now().strftime("%m-%d-%H-%M-%S")),net = net, type = type)