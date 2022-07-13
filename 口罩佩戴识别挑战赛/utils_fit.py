from tqdm import tqdm
import torch
from utils import get_acc
import os

def freeze_net(net, net_name, epoch, freeze_epoch, Dp):
    if epoch < freeze_epoch:
        for param in net.parameters():
            param.requires_grad = False
        try:
            if 'ViT' in net_name or 'ConvNeXt' in net_name or 'BiT' in net_name or 'DeiT' in net_name or 'CaiT' in net_name or 'Swin' in net_name:
                if Dp:
                    for param in net.module.head.parameters():
                        param.requires_grad = True
                else:
                    for param in net.head.parameters():
                        param.requires_grad = True
            elif 'Res' in net_name:
                if Dp:
                    for param in net.module.fc.parameters():
                        param.requires_grad = True
                else:
                    for param in net.fc.parameters():
                        param.requires_grad = True
            else:
                if Dp:
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
        for param in net.parameters():
            param.requires_grad = True


def fit_one_epoch(net, epoch, epochs, trainloader, optimizer, loss_fn, scheduler, early_stopping, cuda, fp16, scaler, save_net, best_acc, checkpoint = './checkpoint'):
    epoch_step = len(trainloader)
    if epoch_step == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集，或者减小batchsize")
    net.train()
    train_loss = 0
    train_acc = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
        for step,(im,label) in enumerate(trainloader,start=0):
            with torch.no_grad():
                if cuda:
                    im = im.cuda()
                    label = label.cuda()
            #---------------------
            #  释放内存
            #---------------------
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            if not fp16:
                #----------------------#
                #   前向传播forward
                #----------------------#
                outputs = net(im)
                #----------------------#
                #   计算损失
                #----------------------#
                loss = loss_fn(outputs,label)
                #----------------------#
                #   反向传播
                #----------------------#
                # backward
                loss.backward()
                # 更新参数
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    #----------------------#
                    #   前向传播forward
                    #----------------------#
                    outputs = net(im)
                    #----------------------#
                    #   计算损失
                    #----------------------#
                    loss = loss_fn(outputs,label)
                #----------------------#
                #   反向传播
                #----------------------#
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            train_loss += loss.data
            train_acc += get_acc(outputs,label)
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
        if not os.path.isdir(checkpoint):
            os.mkdir(checkpoint)
        torch.save(state, '{}/best_{}_ckpt.pth'.format(checkpoint, save_net))
        best_acc = train_acc

    print('Finish Train')
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': train_acc,
        'epoch': epoch + 1,
        'lr': lr,
    }
    torch.save(state, '{}/{}_ckpt.pth'.format(checkpoint, save_net))
    
    early_stopping(train_loss, net)
    # 若满足 early stopping 要求
    if early_stopping.early_stop:
        print("Early stopping")
        # 结束模型训练
        exit()


def fit_one_epoch_test(net, epoch, epochs, testloader, optimizer, loss_fn, scheduler, early_stopping, cuda, fp16, scaler, save_net, best_acc):
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
            with torch.no_grad():
                if cuda:
                    im = im.cuda()
                    label = label.cuda()

                if step >= epoch_step_test:
                    break
                
                # 释放内存
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = net(im)
                loss = loss_fn(outputs,label)
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
        torch.save(state, './checkpoint/{}_ckpt.pth'.format(save_net))
        best_acc = test_acc
        
    print('Finish Test')

    early_stopping(test_loss, net)
    # 若满足 early stopping 要求
    if early_stopping.early_stop:
        print("Early stopping")
        # 结束模型训练
        exit()
