import torch
import torchvision
import torchvision.transforms as transforms


image_mean = [0.4940, 0.4187, 0.3855]
image_std = [0.2048, 0.1941, 0.1932]

# Data
def get_training_dataloader(batch_size = 64, num_workers = 4, shuffle = True, resize = 224, root = './data//'):
    print('==> Preparing Train data..')
    transform_train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(resize, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ])
    trainset = torchvision.datasets.ImageFolder(root + 'train',transform=transform_train)
    classes = trainset.classes
    print("==> 分类类别", " ".join(classes))
    # print(trainset)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers= num_workers, pin_memory = True)
    return trainloader
    
def get_test_dataloader(batch_size = 64, num_workers = 4, shuffle = True, resize = 224, root = './data//'): 
    print('==> Preparing Test data..')   
    transform_test = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ])
    root = './data//'
    testset = torchvision.datasets.ImageFolder(root + 'test',transform= transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=shuffle, num_workers= num_workers, pin_memory = True)
    return testloader

# from utils import get_mean_and_std
# root = './data//'
# transform_test = transforms.Compose([
#         transforms.Resize(224),
#         transforms.ToTensor(),
#     ])
# trainset = torchvision.datasets.ImageFolder(root + 'train',transform_test)
# print(get_mean_and_std(trainset))