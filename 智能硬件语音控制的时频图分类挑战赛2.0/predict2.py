import torch
from torch.utils.data.dataset import Dataset
import time
import numpy as np
import cv2
import pandas as pd
from utils import Get_model, remove_prefix

class XunFeiDataset(Dataset):
    def __init__(self, img_path, label, transform=None):
        self.img_path = img_path
        self.label = label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    def __getitem__(self, index):
        img = cv2.imread(self.img_path[index])            
        img = img.astype(np.float32)
        img /= 255.0
        img -= 1
        if self.transform is not None:
            img = self.transform(image = img)['image']
        img = img.transpose([2,0,1])
        return img,torch.from_numpy(np.array(self.label[index]))
    
    def __len__(self):
        return len(self.img_path)

import albumentations as A 

def predict(test_loader, model):
    model.eval()
    test_pred = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)
            test_pred.append(output.data.cpu().numpy())
            
    return np.vstack(test_pred)

test_df = pd.read_csv('data/submit.csv')
test_df['path'] = 'data/test/' + test_df['image']

test_loader = torch.utils.data.DataLoader(
    XunFeiDataset(test_df['path'].values, [0] * test_df.shape[0],
            A.Compose([
            # A.Resize(512, 512),
            A.RandomCrop(450, 750),
            # A.HorizontalFlip(p=0.5),
            # A.RandomContrast(p=0.5),
        ])
    ), batch_size=10, shuffle=False, num_workers=1, pin_memory=False
)

import glob
def load_model(net = "ResNet18", verbose = True, checkpoint_path = 'checkpoint', num_classes = 24):
    model = Get_model(net, num_classes= num_classes)
    checkpoint = torch.load(r'./{}/{}_ckpt.pth'.format(checkpoint_path ,net))
    acc = checkpoint['acc']
    checkpoint_best = torch.load('./{}/best_{}_ckpt.pth'.format(checkpoint_path ,net))
    best_acc = checkpoint_best['acc']
    if verbose:
        print('==> Resuming from checkpoint..')
        print("我们使用的是 {} 模型进行测试".format(net))
        print('训练时，一共迭代了{}次，最后一次的准确率大概是 {} %'.format(checkpoint['epoch'],acc))
        print('验证时，最佳的准确率的结果为 {} %'.format(best_acc))
    model.load_state_dict(remove_prefix(checkpoint_best['net'], 'module.'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

pred = None
# 重复预测多次test_loader
paths = glob.glob(r'./checkpoint/best*')
for path in paths:
    net = path.split('_')[1]
    model = load_model(net)
    for _ in range(5):
        if pred is None:
            pred = predict(test_loader, model)
        else:
            pred += predict(test_loader, model)
    print("==> {} 模型预测完毕".format(net))

print("==> 正在生成文件")
submit = pd.DataFrame(
    {
        'image': [x.split('/')[-1] for x in test_df['path'].values],
        'label': pred.argmax(1)
})

from datetime import datetime
submit.to_csv('submit_{}_{}.csv'.format(net,datetime.now().strftime("%m-%d-%H-%M-%S")), index=None)