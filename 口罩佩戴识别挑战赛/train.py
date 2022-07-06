'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from utils import Get_model, LabelSmoothCELoss
import os
import argparse
from utils import get_acc,EarlyStopping,remove_prefix
from dataloader import get_training_dataloader
from tqdm import tqdm

# CUDA_VISIBLE_DEVICES=3 python train.py -f --cuda 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--lr', default=0.004, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help =' use GPU?')
    parser.add_argument('--batch-size', default=64, type=int, help = "Batch Size for Training")
    parser.add_argument('--num-workers', default=4, type=int, help = 'num-workers')
    parser.add_argument('--net', type = str, choices=['LeNet5', 'AlexNet', 'VGG16','VGG19',
                                                       'ResNet34','ResNet50','ResNet101',   
                                                       'DenseNet','DenseNet121','DenseNet169','DenseNet201',
                                                       'MobileNetv1','MobileNetv2',
                                                       'ResNeXt50_32x4d','ResNeXt101_32x8d',
                                                       'EfficientNet_b0','EfficientNet_b1','EfficientNet_b2','EfficientNet_b3','EfficientNet_b4','EfficienNet_b5','EfficientNet_b6','EfficientNet_b7','EfficientNet_b8',
                                                       'EfficientNetv2-S','EfficientNetv2-M','EfficientNetv2-L','EfficientNetv2-XL',
                                                       'ConvNeXt-T','ConvNeXt-S','ConvNeXt-B','ConvNeXt-L','ConvNeXt-XL',
                                                       'Swin-M','Swin-L'
                                                       'ViT-B','ViT-L','ViT-H',
                                                       'CaiT_s24','CaiT_xxs24','CaiT_xxs36',
                                                       'DeiT-B','DeiT-T','DeiT-S',
                                                       'BiT-M-resnet152x4','BiT-M-resnet152x2','BiT-M-resnet101x3','BiT-M-resnet101x1'], default='MobileNetv2', help='net type')
    parser.add_argument('--epochs', type = int, default=20, help = 'Epochs')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--patience', '-p', type = int, default=7, help='patience for Early stop')
    parser.add_argument('--optim','-o',type = str, choices = ['sgd','adam','adamw'], default = 'adamw', help = 'choose optimizer')
    parser.add_argument('--resize',type=int,default=224)
    parser.add_argument('-f',action='store_true',help='choose to freeze')
    parser.add_argument('-fe',type=int,default=20)
    parser.add_argument('-dp',action='store_false')

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
    net = Get_model(args.net, num_classes = num_classes)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        
        checkpoint = torch.load('./checkpoint/{}_ckpt.pth'.format(args.net))
        checkpoint_best = torch.load('./checkpoint/best_{}_ckpt.pth'.format(args.net))
        net.load_state_dict(remove_prefix(checkpoint['net'], 'module.'))
        # net.load_state_dict(checkpoint['net'])
        last_acc = checkpoint['acc']
        best_acc = checkpoint_best['acc']
        start_epoch = checkpoint['epoch']
        args.lr = checkpoint['lr']
        print("从EPOCH = {} 开始训练， 学习率为 {} , 最佳的结果ACC为 {:.2f}, 上一次的结果ACC为{}".format(start_epoch + 1,args.lr,best_acc,last_acc))


    if args.cuda:
        device = 'cuda'
        if args.dp:
            net = torch.nn.DataParallel(net)
        net = net.to(device)
        # 当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        
    # print(net)
    # from torchinfo import summary
    # summary(net,(2,3,224,224))
    
    early_stopping = EarlyStopping(patience = args.patience, verbose=True)
    criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothCELoss()
    if args.optim == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.94,verbose=True,patience = 1,min_lr = 0.000001) # 动态更新学习率
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
        
        early_stopping(train_loss, net)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            exit()
   
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

    print("==> 你选择了{}模型，准备开始训练".format(args.net))
    flag = False # 标志只需要做一次操作即可，后续加载数据不需要多次操作
    for epoch in range(start_epoch, epochs):
        if freeze and not flag:
            if epoch < freeze_epoch:
                for param in net.parameters():
                    param.requires_grad = False
                try:
                    if 'ViT' in args.net or 'ConvNeXt' in args.net or 'BiT' in args.net:
                        if args.dp:
                            for param in net.module.head.parameters():
                                param.requires_grad = True
                        else:
                            for param in net.head.parameters():
                                param.requires_grad = True
                    elif 'ResNeXt' in args.net:
                        if args.dp:
                            for param in net.module.fc.parameters():
                                param.requires_grad = True
                        else:
                            for param in net.fc.parameters():
                                param.requires_grad = True
                    else:
                        if args.dp:
                            for param in net.module.classifier.parameters():
                                param.requires_grad = True
                        else:
                            for param in net.classifier.parameters():
                                param.requires_grad = True
                except Exception as e:
                    print(net)
                    print("==>冻结分类层出现错误")
                    print(e)
                    exit()
            else:
                flag = True
                for param in net.parameters():
                    param.requires_grad = True
                trainloader = get_training_dataloader(batch_size = args.batch_size//2, num_workers = args.num_workers, resize = args.resize)
        train(epoch,trainloader)
        # test(epoch,testloader)
        
    torch.cuda.empty_cache()
    