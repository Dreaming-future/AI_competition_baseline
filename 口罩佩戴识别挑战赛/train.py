'''Train CIFAR10 with PyTorch.'''
from unittest import TestLoader
from nets.ConvNeXt import convnext_small
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from utils import get_acc,EarlyStopping
from dataloader import get_test_dataloader, get_training_dataloader
from tqdm import tqdm

# CUDA_VISIBLE_DEVICES=3 python train.py -f --cuda 
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help =' use GPU?')
    parser.add_argument('--batch-size', default=64, type=int, help = "Batch Size for Training")
    parser.add_argument('--num-workers', default=2, type=int, help = 'num-workers')
    parser.add_argument('--net', type = str, choices=['LeNet5', 'AlexNet', 'VGG16','VGG19','ResNet50','ResNet34',   
                                                       'DenseNet','DenseNet121','DenseNet169','DenseNet201',
                                                       'MobileNetv1','MobileNetv2','ResNeXt',
                                                       'ConvNeXt-T','ConvNeXt-S','ConvNeXt-B','ConvNeXt-L','ConvNeXt-XL'], default='MobileNetv2', help='net type')
    parser.add_argument('--epochs', type = int, default=20, help = 'Epochs')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--patience', '-p', type = int, default=7, help='patience for Early stop')
    parser.add_argument('--optim','-o',type = str, choices = ['sgd','adam','adamw'], default = 'adamw', help = 'choose optimizer')
    parser.add_argument('--resize',type=int,default=224)
    parser.add_argument('-f',action='store_true',help='choose to freeze')
    parser.add_argument('-fe',type=int,default=20)

    args = parser.parse_args()
    
    print(args)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    num_classes = 3

    freeze = args.f # 是否冻结训练
    freeze_epoch = args.fe
    # Train Data
    trainloader = get_training_dataloader(batch_size = args.batch_size, num_workers = args.num_workers, resize = args.resize)
    # testloader = get_test_dataloader(batch_size = args.batch_size, num_workers = args.num_workers, shuffle=False, resize = args.resize)


    #  Model
    print('==> Building model..')
    if args.net == 'VGG16':
        from nets.VGG import VGG
        net = VGG('VGG16')
    elif args.net == 'VGG19':
        from nets.VGG import VGG
        net = VGG('VGG19')
    elif args.net == 'ResNet50':
        from nets.ResNet import ResNet50
        net = ResNet50(num_classes)
    elif args.net == 'ResNet34':
        from nets.ResNet import ResNet34
        net = ResNet34(num_classes)
    elif args.net == 'LeNet5':
        from nets.LeNet5 import LeNet5
        net = LeNet5(num_classes)
    elif args.net == 'AlexNet':
        from nets.AlexNet import AlexNet
        net = AlexNet(num_classes)
    elif args.net == 'DenseNet':
        from nets.DenseNet import densenet_cifar
        net = densenet_cifar()
    elif args.net == 'DenseNet121':
        from nets.DenseNet import DenseNet121
        net = DenseNet121(num_classes)
    elif args.net == 'DenseNet169':
        from nets.DenseNet import DenseNet169
        net = DenseNet169(num_classes)
    elif args.net == 'DenseNet201':
        from nets.DenseNet import DenseNet201
        net = DenseNet201(num_classes)
    elif args.net == 'MobileNetv1':
        from nets.MobileNetv1 import MobileNet
        net = MobileNet(num_classes)
    elif args.net == 'MobileNetv2':
        from nets.MobileNetv2 import MobileNetV2
        net = MobileNetV2(num_classes)
    elif args.net == 'ResNeXt':
        from nets.ResNeXt import ResNeXt50
        net = ResNeXt50(num_classes)
    elif args.net == 'ConvNeXt-T':
        from nets.ConvNeXt import convnext_tiny
        net = convnext_tiny(num_classes)
    elif args.net == 'ConvNeXt-S':
        from nets.ConvNeXt import convnext_small
        net = convnext_small(num_classes)
    elif args.net == 'ConvNeXt-B':
        from nets.ConvNeXt import convnext_base
        net = convnext_base(num_classes)
    elif args.net == 'ConvNeXt-L':
        from nets.ConvNeXt import convnext_large
        net = convnext_large(num_classes)
    elif args.net == 'ConvNeXt-XL':
        from nets.ConvNeXt import convnext_xlarge
        net = convnext_xlarge(num_classes)


    if args.cuda:
        device = 'cuda'
        # net = torch.nn.DataParallel(net)
        net = net.to(device)
        # 当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        
    # print(net)
    # from torchinfo import summary
    # summary(net,(2,3,224,224))
    
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        
        checkpoint = torch.load('./checkpoint/{}_ckpt.pth'.format(args.net))
        checkpoint_best = torch.load('./checkpoint/best_{}_ckpt.pth'.format(args.net))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint_best['acc']
        start_epoch = checkpoint['epoch']
        args.lr = checkpoint['lr']
        print("从{} 开始训练， 学习率为 {} , 最佳的结果ACC为{}".format(start_epoch + 1,args.lr,best_acc))

    early_stopping = EarlyStopping(patience = args.patience, verbose=True)
    criterion = nn.CrossEntropyLoss()
    if args.optim == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.94,verbose=True,patience = 1,min_lr = 0.000001) # 动态更新学习率
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30,40], gamma=0.1)
    epochs = args.epochs
    def train(epoch,trainloader):
        epoch_step = len(trainloader)
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集，或者减小batchsize")
        net.train()
        train_loss = 0
        train_acc = 0
        global best_acc
        print('Start Train')
        with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
            for step,(im,label) in enumerate(trainloader,start=0):
                im = im.to(device)
                label = label.to(device)
                #---------------------
                #  释放内存
                #---------------------
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播forward
                #----------------------#
                outputs = net(im)
                #----------------------#
                #   计算损失
                #----------------------#
                loss = criterion(outputs,label)
                train_loss += loss.data
                train_acc += get_acc(outputs,label)
                #----------------------#
                #   反向传播
                #----------------------#
                # backward
                loss.backward()
                # 更新参数
                optimizer.step()
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(**{'Train Loss' : train_loss.item()/(step+1),
                                    'Train Acc' :train_acc.item()/(step+1),  
                                    'Lr'   : lr})
                pbar.update(1)
        # train_loss = train_loss.item() / len(trainloader)
        train_acc = train_acc.item() * 100 / len(trainloader)
        # scheduler.step(train_loss)
        scheduler.step()
        
        # Save checkpoint.
        if train_acc > best_acc:
            print('Saving Best Model..')
            state = {
                'net': net.state_dict(),
                'acc': train_acc,
                'epoch': epoch + 1,
                'lr': lr,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/best_{}_ckpt.pth'.format(args.net))
            best_acc = train_acc
    
        print('Finish Train')
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': train_acc,
            'epoch': epoch + 1,
            'lr': lr,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}_ckpt.pth'.format(args.net))

    def test(epoch,testloader):
        global best_acc
        epoch_step_test = len(testloader)
        if epoch_step_test == 0:
                raise ValueError("数据集过小，无法进行训练，请扩充数据集，或者减小batchsize")
        
        net.eval()
        test_loss = 0
        test_acc = 0
        print('Start Test')
        #--------------------------------
        #   相同方法，同train
        #--------------------------------
        with tqdm(total=epoch_step_test,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar2:
            for step,(im,label) in enumerate(testloader,start=0):
                im = im.to(device)
                label = label.to(device)
                with torch.no_grad():
                    if step >= epoch_step_test:
                        break
                    
                    # 释放内存
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    #----------------------#
                    #   前向传播
                    #----------------------#
                    outputs = net(im)
                    loss = criterion(outputs,label)
                    test_loss += loss.data
                    test_acc += get_acc(outputs,label)
                    
                    pbar2.set_postfix(**{'Test Acc': test_acc.item()/(step+1),
                                'Test Loss': test_loss.item() / (step + 1)})
                    pbar2.update(1)
        lr = optimizer.param_groups[0]['lr']
        test_acc = test_acc.item() * 100 / len(testloader)
        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
                'lr': lr,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/{}_ckpt.pth'.format(args.net))
            best_acc = test_acc
            
        print('Finish Test')

        early_stopping(test_loss, net)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            exit()
    
    flag = False # 标志只需要做一次操作即可，后续加载数据不需要多次操作
    for epoch in range(start_epoch, epochs):
        if freeze and not flag:
            if epoch < freeze_epoch:
                for param in net.parameters():
                    param.requires_grad = False

                for param in net.head.parameters():
                    param.requires_grad = True
            else:
                flag = True
                for param in net.parameters():
                    param.requires_grad = True
                trainloader = get_training_dataloader(batch_size = args.batch_size//2, num_workers = args.num_workers, resize = args.resize)
        train(epoch,trainloader)
        # test(epoch,testloader)
        
    torch.cuda.empty_cache()
    