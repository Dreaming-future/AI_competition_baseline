import torch
import torchvision
import torchvision.transforms as transforms


image_mean = [0.4940, 0.4187, 0.3855]
image_std = [0.2048, 0.1941, 0.1932]


def get_train_val_split_data( train_set, train_split = 0.8):
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(num_train * train_split)
    train_idx, val_idx = indices[:split], indices[split:]
    train_dataset = torch.utils.data.Subset(train_set,train_idx)
    val_dataset = torch.utils.data.Subset(train_set, val_idx)
    return train_dataset, val_dataset

# Data
def get_training_val_dataloader(batch_size = 64, num_workers = 4, shuffle = True, resize = 224, root = './data//'):
    print('==> Preparing Train data..')
    transform_train = transforms.Compose([
        transforms.Resize(resize),
        # transforms.RandomCrop(resize, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomInvert(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotations(30),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ])
    root = './data//'
    trainset = torchvision.datasets.ImageFolder(root + 'train',transform=transform_train)
    classes = trainset.classes
    print("==> 分类类别", " ".join(classes))
    train_dataset, val_dataset = get_train_val_split_data(trainset, train_split= 0.9)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers= num_workers, pin_memory = True)
    
    return train_loader
    
def get_val_dataloader(batch_size = 64, num_workers = 4, shuffle = False, resize = 224, root = './data//'): 
    print('==> Preparing Test data..')   
    transform_test = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ])
    root = './data//'
    valset = torchvision.datasets.ImageFolder(root + 'train',transform= transform_test)
    train_dataset, val_dataset = get_train_val_split_data(valset, train_split= 0.9)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers= num_workers, pin_memory = True)
    return val_loader

# from utils import get_mean_and_std
# root = './data//'
# transform_test = transforms.Compose([
#         transforms.Resize(224),
#         transforms.ToTensor(),
#     ])
# trainset = torchvision.datasets.ImageFolder(root + 'train',transform_test)
# print(get_mean_and_std(trainset))