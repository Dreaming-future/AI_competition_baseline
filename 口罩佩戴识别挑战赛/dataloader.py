import torch
import torchvision
import torchvision.transforms as transforms

# Data
def get_training_dataloader(batch_size = 64, num_workers = 4, shuffle = True, resize = 224):
    print('==> Preparing Train data..')
    transform_train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(resize, padding=4),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    root = './data//'
    trainset = torchvision.datasets.ImageFolder(root + 'train',transform=transform_train)
    # classes = trainset.classes
    # print("分类类别",classes)
    # print(trainset)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers= num_workers)
    return trainloader
    
def get_test_dataloader(batch_size = 64, num_workers = 4, shuffle = True, resize = 224): 
    print('==> Preparing Test data..')   
    transform_test = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    root = './data//'
    testset = torchvision.datasets.ImageFolder(root + 'test',transform= transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=shuffle, num_workers= num_workers)
    return testloader

