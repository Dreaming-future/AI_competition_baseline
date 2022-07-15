import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset

train_df = pd.read_csv('./data/train.csv')
train_df['path'] = './data/train/' + train_df['image']
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
# Data
def get_training_dataloader(batch_size = 64, num_workers = 4, shuffle = True, resize = 224, root = './data//'):
    print('==> Preparing Train data..')
    transform_train = A.Compose([
            # A.Resize(512, 512),
            A.RandomCrop(450, 750),
            # A.HorizontalFlip(p=0.5),
            # A.RandomContrast(p=0.5),
            A.CoarseDropout(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=0, p=0.5),
            # A.HueSaturationValue(p=0.5),
            # A.RandomBrightnessContrast(p=0.5),
        ])
    
    trainset = XunFeiDataset(train_df['path'].values[:-200], train_df['label'].values[:-200], transform= transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers= num_workers, pin_memory = True)
    return trainloader
    
def get_val_dataloader(batch_size = 64, num_workers = 4, shuffle = True, resize = 224, root = './data//'): 
    print('==> Preparing Val data..')   
    transform_test = A.Compose([
            # A.Resize(512, 512),
            A.RandomCrop(450, 750),
            # A.HorizontalFlip(p=0.5),
            # A.RandomContrast(p=0.5),
        ])
    valset = XunFeiDataset(train_df['path'].values[-200:], train_df['label'].values[-200:], transform= transform_test)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=shuffle, num_workers= num_workers, pin_memory = True)
    return val_loader
