'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from utils import Get_model, LabelSmoothCELoss, empty_cache
import os
import argparse
from utils import get_acc,EarlyStopping,remove_prefix
from dataloader import get_training_dataloader, get_val_dataloader
from tqdm import tqdm
from utils_fit import fit_one_epoch, freeze_net, fit_one_epoch_val

# CUDA_VISIBLE_DEVICES=3 python train.py -f --cuda 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--lr','-lr', default=0.004, type=float, help='learning rate')
    parser.add_argument('--num-classes','-nc', default=24, type=int, help='learning rate')
    parser.add_argument('--cuda', '-gpu', action='store_true', default=False, help =' use GPU?')
    parser.add_argument('--batch-size','-bs', default=32, type=int, help = "Batch Size for Training")
    parser.add_argument('--num-workers', '-nw', default=8, type=int, help = 'num-workers')
    parser.add_argument('--net', '--model', type = str, choices=['LeNet5', 'AlexNet', 'VGG16','VGG19',
                                                       'ResNet18','ResNet34','ResNet50','ResNet101',   
                                                       'DenseNet','DenseNet121','DenseNet161','DenseNet169','DenseNet201',
                                                       'MobileNetv1','MobileNetv2',
                                                       'HRNet-w18','HRNet-w18-S','HRNet-w18-Sv2','HRNet-w30','HRNet-w32','HRNet-w40','HRNet-w44','HRNet-w48','HRNet-w64',
                                                       'ResNeXt50-32x4d','ResNeXt101-32x8d',
                                                       'EfficientNet-b0','EfficientNet-b1','EfficientNet-b2','EfficientNet-b3','EfficientNet-b4','EfficienNet-b5','EfficientNet-b6','EfficientNet-b7','EfficientNet-b8',
                                                       'Efficientv2-b0','Efficientv2-b1','Efficientv2-b2','Efficientv2-b3',
                                                       'Efficientv2-T','Efficientv2-S','Efficientv2-M','Efficientv2-L','Efficientv2-XL',
                                                       'ConvNeXt-T','ConvNeXt-S','ConvNeXt-B','ConvNeXt-L','ConvNeXt-XL',
                                                       'Swin-T','Swin-S','Swin-B','Swin-L',
                                                       'ViT-B','ViT-L','ViT-H',
                                                       'CaiT-s24','CaiT-xxs24','CaiT-xxs36',
                                                       'DeiT-T','DeiT-S','DeiT-B',
                                                       'BiT-M-resnet152x4','BiT-M-resnet152x2','BiT-M-resnet101x3','BiT-M-resnet101x1'], default='MobileNetv2', help='net type')
    parser.add_argument('--epochs', '-e', type = int, default=20, help = 'Epochs')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint 断点续传')
    parser.add_argument('--resume-lr','-rlr',type = float, default=1, help = '断点训练时的学习率是否改变')
    parser.add_argument('--patience', '-p', type = int, default=7, help='patience for Early stop')
    parser.add_argument('--optim','-o',type = str, choices = ['sgd','adam','adamw'], default = 'adamw', help = 'choose optimizer')
    parser.add_argument('--resize','-rs', type=int,default=224, help = '图像的shape')
    parser.add_argument('--f','-f',action='store_true',help='choose to freeze 是否使用冻结训练')
    parser.add_argument('--fe','-fe',type=int,default=15, help = '冻结训练的迭代次数')
    parser.add_argument('--dp','-dp',action='store_false', help = '是否使用并行训练，多GPU')
    parser.add_argument('--fp16','-fp16',action='store_true',help='是否使用混合精度训练')
    parser.add_argument('--data', default='./data//', type = str, help = '数据集路径')
    parser.add_argument('--checkpoint', '-ck', type = str, default='checkpoint', help = '保存模型')
    args = parser.parse_args()
    print(args)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    num_classes = args.num_classes
    freeze = args.f # 是否冻结训练
    if freeze:
        freeze_epoch = args.fe
    else:
        freeze_epoch = 0
    epochs = args.epochs
    fp16 = args.fp16
    resume = args.resume
    lr = args.lr
    Cuda = args.cuda
    Dp = args.dp
    patience = args.patience
    epochs = args.epochs
    Net = args.net
    data = args.data
    checkpoint_model = args.checkpoint
    # Train Data
    train_loader = get_training_dataloader(batch_size = args.batch_size, num_workers = args.num_workers, resize = args.resize, root=data)
    val_loader = get_val_dataloader(batch_size = args.batch_size*2, num_workers = args.num_workers, shuffle=False, resize = args.resize)

    #  Model
    print('==> Building model..')
    net = Get_model(Net, num_classes = num_classes)
    if args.resume_lr != 1:
        resume = True
    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(checkpoint_model), 'Error: no checkpoint directory found!'

        checkpoint_best = torch.load('{}/best_{}_ckpt.pth'.format(checkpoint_model, Net))
        best_acc = checkpoint_best['acc']
        empty_cache()
        checkpoint = torch.load('{}/{}_ckpt.pth'.format(checkpoint_model, Net))
        net.load_state_dict(remove_prefix(checkpoint['net'], 'module.'))

        last_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        if args.resume_lr != 1:
            lr = args.resume_lr
        print("从EPOCH = {} 开始训练， 学习率为 {} , 最佳的结果ACC为 {:.2f}, 上一次的结果ACC为{}".format(start_epoch + 1, lr, best_acc, last_acc))

    #------------------------------------#
    #  使用GPU，并且判断是否利用并行训练
    #------------------------------------#
    if Cuda:
        device = 'cuda'
        if Dp:
            print("==> 开启并行模式")
            net = torch.nn.DataParallel(net)
        net = net.to(device)
        # 当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        
    
    if fp16:
        #------------------------------------------------------------------#
        #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
        #   因此torch1.2这里显示"could not be resolve"
        #------------------------------------------------------------------#
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
        print('==> 使用混合精度的方法训练')
    else:
        scaler = None

    early_stopping = EarlyStopping(patience = patience, verbose=True)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = LabelSmoothCELoss()
    if args.optim == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr= lr)
    elif args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr= lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr= lr,
                        momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.94,verbose=True,patience = 1,min_lr = 0.000001) # 动态更新学习率
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= epochs)

    print("==> 你选择了{}模型，准备开始训练".format(Net))

    for epoch in range(start_epoch, epochs):
        freeze_net(net, Net , epoch, freeze_epoch, Dp)
        fit_one_epoch(net, epoch, freeze_epoch, epochs, train_loader, optimizer, loss_fn, scheduler, early_stopping, Cuda, fp16, scaler, Net, best_acc, checkpoint = checkpoint_model)
        fit_one_epoch_val(net, epoch, freeze_epoch, epochs, val_loader, optimizer, loss_fn, scheduler, early_stopping, Cuda, fp16, scaler, Net, best_acc, checkpoint = checkpoint_model)
    torch.cuda.empty_cache()