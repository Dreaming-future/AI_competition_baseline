import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


def empty_cache():
    if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
def Get_model(net, num_classes = 1000, verbose = False):
    if net == 'VGG16':
        from nets.VGG import VGG16_bn
        model = VGG16_bn(num_classes)
    elif net == 'VGG19':
        from nets.VGG import VGG19_bn
        model = VGG19_bn(num_classes)
    elif net == "ResNet18":
        from nets.ResNet import ResNet18
        model = ResNet18(num_classes)
    elif net == 'ResNet34':
        from nets.ResNet import ResNet34
        model = ResNet34(num_classes)
    elif net == 'ResNet50':
        from nets.ResNet import ResNet50
        model = ResNet50(num_classes)
    elif net == 'ResNet101':
        from nets.ResNet import ResNet101
        model = ResNet101(num_classes)
    elif net == 'LeNet5':
        from nets.LeNet5 import LeNet5
        model = LeNet5(num_classes)
    elif net == 'AlexNet':
        from nets.AlexNet import AlexNet
        model = AlexNet(num_classes)
    elif net == 'DenseNet':
        from nets.DenseNet import densenet_cifar
        model = densenet_cifar()
    elif net == 'DenseNet121':
        from nets.DenseNet import DenseNet121
        model = DenseNet121(num_classes)
    elif net == 'DenseNet161':
        from nets.DenseNet import DenseNet161
        model = DenseNet161(num_classes)
    elif net == 'DenseNet169':
        from nets.DenseNet import DenseNet169
        model = DenseNet169(num_classes)
    elif net == 'DenseNet201':
        from nets.DenseNet import DenseNet201
        model = DenseNet201(num_classes)
    elif net == 'MobileNetv1':
        from nets.MobileNetv1 import MobileNet
        model = MobileNet(num_classes)
    elif net == 'MobileNetv2':
        from nets.MobileNetv2 import MobileNetV2
        model = MobileNetV2(num_classes)
    elif net == 'HRNet-w18':
        from nets.HRNet import hrnet_w18
        model = hrnet_w18(num_classes)
    elif net == 'HRNet-w18-S':
        from nets.HRNet import hrnet_w18_small
        model = hrnet_w18_small(num_classes)
    elif net == 'HRNet-w18-Sv2':
        from nets.HRNet import hrnet_w18_small_v2
        model = hrnet_w18_small_v2(num_classes)
    elif net == 'HRNet-w30':
        from nets.HRNet import hrnet_w30
        model = hrnet_w30(num_classes)
    elif net == 'HRNet-w32':
        from nets.HRNet import hrnet_w32
        model = hrnet_w32(num_classes)
    elif net == 'HRNet-w40':
        from nets.HRNet import hrnet_w40
        model = hrnet_w40(num_classes)
    elif net == 'HRNet-w48':
        from nets.HRNet import hrnet_w48
        model = hrnet_w48(num_classes)
    elif net == 'HRNet-w64':
        from nets.HRNet import hrnet_w64
        model = hrnet_w64(num_classes)
    elif net == 'ResNeXt50-32x4d':
        from nets.ResNeXt import ResNeXt50_32x4d
        model = ResNeXt50_32x4d(num_classes)
    elif net == 'ResNeXt101-32x8d':
        from nets.ResNeXt import ResNeXt101_32x8d
        model = ResNeXt101_32x8d(num_classes)
    elif net == 'EfficientNet-b0':
        from nets.EfficientNet import EfficientNet_b0
        model = EfficientNet_b0(num_classes)
    elif net == 'EfficientNet-b1':
        from nets.EfficientNet import EfficientNet_b1
        model = EfficientNet_b1(num_classes)
    elif net == 'EfficientNet-b2':
        from nets.EfficientNet import EfficientNet_b2
        model = EfficientNet_b2(num_classes)
    elif net == 'EfficientNet-b3':
        from nets.EfficientNet import EfficientNet_b3
        model = EfficientNet_b3(num_classes)
    elif net == 'EfficientNet-b4':
        from nets.EfficientNet import EfficientNet_b4
        model = EfficientNet_b4(num_classes)
    elif net == 'EfficientNet-b5':
        from nets.EfficientNet import EfficientNet_b5
        model = EfficientNet_b5(num_classes)
    elif net == 'EfficientNet_b6':
        from nets.EfficientNet import EfficientNet_b6
        model = EfficientNet_b6(num_classes)
    elif net == 'EfficientNet-b7':
        from nets.EfficientNet import EfficientNet_b7
        model = EfficientNet_b7(num_classes)
    elif net == 'EfficientNet-b8':
        from nets.EfficientNet import EfficientNet_b8
        model = EfficientNet_b8(num_classes)
    elif net == 'Efficientv2-b0':
        from nets.EfficientNetv2 import Efficientv2_b0
        model = Efficientv2_b0(num_classes)
    elif net == 'Efficientv2-b1':
        from nets.EfficientNetv2 import Efficientv2_b1
        model = Efficientv2_b1(num_classes)
    elif net == 'Efficientv2-b2':
        from nets.EfficientNetv2 import Efficientv2_b2
        model = Efficientv2_b2(num_classes)
    elif net == 'Efficientv2-b3':
        from nets.EfficientNetv2 import Efficientv2_b3
        model = Efficientv2_b3(num_classes)
    elif net == 'Efficientv2-T':
        from nets.EfficientNetv2 import Efficientv2_T
        model = Efficientv2_T(num_classes)
    elif net == 'Efficientv2-S':
        from nets.EfficientNetv2 import Efficientv2_S
        model = Efficientv2_S(num_classes)
    elif net == 'Efficientv2-M':
        from nets.EfficientNetv2 import Efficientv2_M
        model = Efficientv2_M(num_classes)
    elif net == 'Efficientv2-L':
        from nets.EfficientNetv2 import Efficientv2_L
        model = Efficientv2_L(num_classes)
    elif net == 'Efficientv2-XL':
        from nets.EfficientNetv2 import Efficientv2_XL
        model = Efficientv2_XL(num_classes)
    elif net == 'ConvNeXt-T':
        from nets.ConvNeXt import convnext_tiny
        model = convnext_tiny(num_classes)
    elif net == 'ConvNeXt-S':
        from nets.ConvNeXt import convnext_small
        model = convnext_small(num_classes)
    elif net == 'ConvNeXt-B':
        from nets.ConvNeXt import convnext_base
        model = convnext_base(num_classes)
    elif net == 'ConvNeXt-L':
        from nets.ConvNeXt import convnext_large
        model = convnext_large(num_classes)
    elif net == 'ConvNeXt-XL':
        from nets.ConvNeXt import convnext_xlarge
        model = convnext_xlarge(num_classes)
    elif net == 'ViT-B':
        from nets.ViT import Vit_bash_patch16_224
        model = Vit_bash_patch16_224(num_classes)
    elif net == 'ViT-L':
        from nets.ViT import Vit_large_patch16_224
        model = Vit_large_patch16_224(num_classes)
    elif net == 'ViT-H':
        from nets.ViT import Vit_huge_patch14_224
        model = Vit_huge_patch14_224(num_classes)
    elif net == 'Swin-T':
        from nets.Swin import swin_tiny_patch4_window7_224
        model = swin_tiny_patch4_window7_224(num_classes)
    elif net == 'Swin-S':
        from nets.Swin import swin_small_patch4_window7_224
        model = swin_small_patch4_window7_224(num_classes)
    elif net == 'Swin-B':
        from nets.Swin import swin_base_patch4_window7_224
        model = swin_base_patch4_window7_224(num_classes)
    elif net == 'Swin-L':
        from nets.Swin import swin_large_patch4_window7_224
        model = swin_large_patch4_window7_224(num_classes)
    elif net == 'CaiT-s24':
        from nets.CaiT import CaiT_s24
        model = CaiT_s24(num_classes)
    elif net == 'CaiT-xxs24':
        from nets.CaiT import CaiT_xxs24
        model = CaiT_xxs24(num_classes)
    elif net == 'CaiT-xxs36':
        from nets.CaiT import CaiT_xxs36
        model = CaiT_xxs36(num_classes)
    elif net == 'DeiT-B':
        from nets.DeiT import DeiT_B
        model = DeiT_B(num_classes)
    elif net == 'DeiT-T':
        from nets.DeiT import DeiT_T
        model = DeiT_T(num_classes)
    elif net == 'DeiT-S':
        from nets.DeiT import DeiT_S
        model = DeiT_S(num_classes)
    elif net == 'BiT-M-resnet152x4':
        from nets.BiT import BiT_M_resnet152x4
        model = BiT_M_resnet152x4(num_classes)
    elif net == 'BiT-M-resnet152x2':
        from nets.BiT import BiT_M_resnet152x2
        model = BiT_M_resnet152x2(num_classes)
    elif net == 'BiT-M-resnet101x3':
        from nets.BiT import BiT_M_resnet101x3
        model = BiT_M_resnet101x3(num_classes)
    elif net == 'BiT-M-resnet101x1':
        from nets.BiT import BiT_M_resnet101x1
        model = BiT_M_resnet101x1(num_classes)
    if verbose:
        from torchinfo import summary
        summary(model,(2,3,224,224))
    return model

def remove_prefix(state_dict, prefix):
    '''
    Old style model is stored with all names of parameters
    share common prefix 'module.' 
    '''
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def eval_top1(outputs, label):
    total = outputs.shape[0]
    outputs = torch.softmax(outputs, dim=-1)
    _, pred_y = outputs.data.max(dim=1) # ????????????
    correct = (pred_y == label).sum().data
    return correct / total

def eval_top5(outputs, label):
    total = outputs.shape[0]
    outputs = torch.softmax(outputs, dim=-1)
    pred_y = np.argsort(-outputs.cpu().numpy())
    pred_y_top5 = pred_y[:,:5]
    correct = 0
    for i in range(total):
        if label[i].cpu().numpy() in pred_y_top5[i]:
            correct += 1
    return correct / total

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_acc(outputs, label):
    total = outputs.shape[0]
    probs, pred_y = outputs.data.max(dim=1) # ????????????
    correct = (pred_y == label).sum().data
    return correct / total

'''
????????????
'''
class LabelSmoothCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, smoothing=0.1):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (
            1.0 - smoothing) * one_hot_label + smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()

        return loss

'''
Focal Loss
'''
class focal_loss(nn.Module):    
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss????????????, -??(1-yi)**?? *ce_loss(xi,yi)      
        ???????????????????????? focal_loss????????????.
        :param alpha:   ???????????,????????????.      ?????????????????,??????????????????,?????????????????,???????????????[??, 1-??, 1-??, ....],????????? ???????????????????????????????????? , retainnet????????????0.25
        :param gamma:   ????????,????????????????????????. retainnet????????????2
        :param num_classes:     ????????????
        :param size_average:    ??????????????????,???????????????
        """

        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # ???????????list????????????,size:[num_classes] ??????????????????????????????????????????
            print("Focal_loss alpha = {}, ??????????????????????????????????????????".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #???????????????????????,???????????????????????????,??????????????????????????????
            print(" --- Focal_loss alpha = {} ,???????????????????????????,????????????????????????????????? --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # ?? ????????? [ ??, 1-??, 1-??, 1-??, 1-??, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss????????????        
        :param preds:   ????????????. size:[B,N,C] or [B,C]    ????????????????????????????????????, B ??????, N????????????, C?????????        
        :param labels:  ????????????. size:[B,N] or [B]        
        :return:
        """        
        # assert preds.dim()==2 and labels.dim()==1        
        preds = preds.view(-1,preds.size(-1))        
        self.alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) # ???????????????????????????log_softmax, ?????????????????????softmax?????????(????????????????????????log_softmax,????????????exp??????)        
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # ???????????????nll_loss ( crossempty = log_softmax + nll )        
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        self.alpha = self.alpha.gather(0,labels.view(-1))        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) ???focal loss??? (1-pt)**??
        loss = torch.mul(self.alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')	# ??????????????????????????????????????????
        self.val_loss_min = val_loss

def train(net, trainloader, testloader, epoches, optimizer , criterion, scheduler , path = './model.pth', writer = None ,verbose = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    train_acc_list, test_acc_list = [],[]
    train_loss_list, test_loss_list = [],[]
    lr_list  = []
    for i in range(epoches):
        start = time.time()
        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0
        if torch.cuda.is_available():
            net = net.to(device)
        net.train()
        for step,data in enumerate(trainloader,start=0):
            im,label = data
            im = im.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            # ????????????
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            # formard
            outputs = net(im)
            loss = criterion(outputs,label)
            # backward
            loss.backward()
            # ????????????
            optimizer.step()

            train_loss += loss.data
            # probs, pred_y = outputs.data.max(dim=1) # ????????????
            # # ???????????????
            # train_acc += (pred_y==label).sum().item()
            # # ??????
            # total += label.size(0)
            train_acc += get_acc(outputs,label)
            # ??????????????????
            rate = (step + 1) / len(trainloader)
            a = "*" * int(rate * 50)
            b = "." * (50 - int(rate * 50))
            print('\r train {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(i+1,epoches,int(rate*100),a,b),end='')
        train_loss = train_loss / len(trainloader)
        train_acc = train_acc * 100 / len(trainloader)
        if verbose:
            train_acc_list.append(train_acc.item())
            train_loss_list.append(train_loss.item())
    #     print('train_loss:{:.6f} train_acc:{:3.2f}%' .format(train_loss ,train_acc),end=' ')  
        # ???????????????
        lr = optimizer.param_groups[0]['lr']
        if verbose:
            lr_list.append(lr)
        # ???????????????
        scheduler.step(train_loss)
        if testloader is not None:
            net.eval()
            with torch.no_grad():
                for step,data in enumerate(testloader,start=0):
                    im,label = data
                    im = im.to(device)
                    label = label.to(device)
                    # ????????????
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    outputs = net(im)
                    loss = criterion(outputs,label)
                    test_loss += loss.data
                    # probs, pred_y = outputs.data.max(dim=1) # ????????????
                    # test_acc += (pred_y==label).sum().item()
                    # total += label.size(0)
                    test_acc += get_acc(outputs,label)
                    rate = (step + 1) / len(testloader)
                    a = "*" * int(rate * 50)
                    b = "." * (50 - int(rate * 50))
                    print('\r test  {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(i+1,epoches,int(rate*100),a,b),end='')
            test_loss = test_loss / len(testloader)
            test_acc = test_acc * 100 / len(testloader)
            if verbose:
                test_loss_list.append(test_loss.item())
                test_acc_list.append(test_acc.item())
            end = time.time()
            print(
                '\rEpoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}% Test Loss:{:>.6f}  Test Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(
                    i + 1, epoches, train_loss, train_acc, test_loss, test_acc,lr), end='')
        else:
            end = time.time()
            print('\rEpoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(i+1,epoches,train_loss,train_acc,lr),end = '')
        time_ = int(end - start)
        h = time_ / 3600
        m = time_ % 3600 /60
        s = time_ % 60
        time_str = "\tTime %02d:%02d" % ( m, s)
        # ====================== ?????? tensorboard ==================
        if writer is not None:
            writer.add_scalars('Loss', {'train': train_loss,
                                    'valid': test_loss}, i+1)
            writer.add_scalars('Acc', {'train': train_acc ,
                                   'valid': test_acc}, i+1)
            writer.add_scalars('Learning Rate',lr,i+1)
        # =========================================================
        # ??????????????????
        print(time_str)
        # ????????????????????????????????????????????????
        if test_acc > best_acc:
            torch.save(net,path)
            best_acc = test_acc
    Acc = {}
    Loss = {}
    Acc['train_acc'] = train_acc_list
    Acc['test_acc'] = test_acc_list
    Loss['train_loss'] = train_loss_list
    Loss['test_loss'] = test_loss_list
    Lr = lr_list
    return Acc, Loss, Lr

def train2(net, trainloader, testloader, epoches, optimizer , criterion, scheduler , path = './model.pth', verbose = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    train_acc_list, test_acc_list = [],[]
    train_loss_list, test_loss_list = [],[]
    lr_list  = []
    for i in range(epoches):
        start = time.time()
        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0
        if torch.cuda.is_available():
            net = net.to(device)
        net.train()
        for step,data in enumerate(trainloader,start=0):
            im,label = data
            im = im.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            # ????????????
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            # formard
            outputs = net(im)
            loss = criterion(outputs,label)
            # backward
            loss.backward()
            # ????????????
            optimizer.step()

            train_loss += loss.data
            # probs, pred_y = outputs.data.max(dim=1) # ????????????
            # # ???????????????
            # train_acc += (pred_y==label).sum().item()
            # # ??????
            # total += label.size(0)
            train_acc += get_acc(outputs,label)
            # ??????????????????
            rate = (step + 1) / len(trainloader)
            a = "*" * int(rate * 50)
            b = "." * (50 - int(rate * 50))
            print('\r train {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(i+1,epoches,int(rate*100),a,b),end='')
        train_loss = train_loss / len(trainloader)
        train_acc = train_acc * 100 / len(trainloader)
        if verbose:
            train_acc_list.append(train_acc.item())
            train_loss_list.append(train_loss.item())
    #     print('train_loss:{:.6f} train_acc:{:3.2f}%' .format(train_loss ,train_acc),end=' ')  
        # ???????????????
        lr = optimizer.param_groups[0]['lr']
        if verbose:
            lr_list.append(lr)
        # ???????????????
        scheduler.step()
        if testloader is not None:
            net.eval()
            with torch.no_grad():
                for step,data in enumerate(testloader,start=0):
                    im,label = data
                    im = im.to(device)
                    label = label.to(device)
                    # ????????????
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    outputs = net(im)
                    loss = criterion(outputs,label)
                    test_loss += loss.data
                    # probs, pred_y = outputs.data.max(dim=1) # ????????????
                    # test_acc += (pred_y==label).sum().item()
                    # total += label.size(0)
                    test_acc += get_acc(outputs,label)
                    rate = (step + 1) / len(testloader)
                    a = "*" * int(rate * 50)
                    b = "." * (50 - int(rate * 50))
                    print('\r test  {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(i+1,epoches,int(rate*100),a,b),end='')
            test_loss = test_loss / len(testloader)
            test_acc = test_acc * 100 / len(testloader)
            if verbose:
                test_loss_list.append(test_loss.item())
                test_acc_list.append(test_acc.item())
            end = time.time()
            print(
                '\rEpoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}% Test Loss:{:>.6f}  Test Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(
                    i + 1, epoches, train_loss, train_acc, test_loss, test_acc,lr), end='')
        else:
            end = time.time()
            print('\rEpoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(i+1,epoches,train_loss,train_acc,lr),end = '')
        time_ = int(end - start)
        h = time_ / 3600
        m = time_ % 3600 /60
        s = time_ % 60
        time_str = "\tTime %02d:%02d" % ( m, s)
        # ??????????????????
        print(time_str)
        # ????????????????????????????????????????????????
        if test_acc > best_acc:
            torch.save(net,path)
            best_acc = test_acc
    Acc = {}
    Loss = {}
    Acc['train_acc'] = train_acc_list
    Acc['test_acc'] = test_acc_list
    Loss['train_loss'] = train_loss_list
    Loss['test_loss'] = test_loss_list
    Lr = lr_list
    return Acc, Loss, Lr

def plot_history(epoches, Acc, Loss, lr):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    epoch_list = range(1,epoches + 1)
    plt.plot(epoch_list, Loss['train_loss'])
    plt.plot(epoch_list, Loss['test_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss Value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(epoch_list, Acc['train_acc'])
    plt.plot(epoch_list, Acc['test_acc'])
    plt.xlabel('epoch')
    plt.ylabel('Acc Value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(epoch_list, lr)
    plt.xlabel('epoch')
    plt.ylabel('Train LR')
    plt.show()

