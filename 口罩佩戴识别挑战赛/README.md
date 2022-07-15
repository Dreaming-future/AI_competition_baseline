# 口罩佩戴识别挑战赛

比赛地址：http://challenge.xfyun.cn/topic/info?type=wear-mask&option=tjjg

## 一、赛事背景

日常生活中，面对经呼吸道传播的包括新冠等在内的传染性疾病，人们佩戴口罩进行防护可保护身体健康和生命安全。在公共交通、密闭场所、人员密集场所，以及商场点单、购物、收取快递外卖等情况下，都应规范戴好口罩。

口罩佩戴时要分清正反，盖住口鼻和下巴，鼻夹要压实，出现脏污、变形、损坏、异味需要及时更换，连续佩戴时间不超过8小时。人脸佩戴口罩的自动化识别可以有效监督人们是否规范佩戴口罩，是抑制疾病在人流量大的公共场合快速传播和保护身体健康的重要技术手段。

![img](https://img-blog.csdnimg.cn/img_convert/dfea2b09592671cfcf981750ad2aaae7.png)

## 二、赛事任务

本次赛题需要选手对口罩是否正确佩戴进行分类，训练集和测试集的图片已经通过人脸识别，只需要选手完成分类任务。我们将口罩佩戴的状态分为三类：

mask_weared_incorrect：口罩佩戴不正确（漏出鼻子或者嘴巴）

![img](https://img-blog.csdnimg.cn/img_convert/e7f73c7a627c609bb8bb2e07c4e2e899.png)

with_mask：正确佩戴口罩

![img](https://img-blog.csdnimg.cn/img_convert/33465c4c351da897214921687161ad3e.png)

without_mask：没有佩戴口罩

![img](https://img-blog.csdnimg.cn/img_convert/889b53a39695816b9edc2fb01548d4b8.png)

选手需要根据训练集构建模型对测试集的图片进行预测，按照测试集精度进行排名。

## 三、评审规则

### 1. 数据说明

赛题数据由训练集和测试集组成，训练数据集按照口罩佩戴的不同类型进行存放。

### 2. 评估指标

本次竞赛的评价标准采用准确率指标，最高分为1。

计算方法参考：

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

评估代码参考：

![img](https://img-blog.csdnimg.cn/img_convert/cb8833e0c3851cf927e442395d9d6064.png)

### 3. 评测及排行

1、 赛事提供下载数据，选手在本地进行算法调试，在比赛页面提交结果。

2、 每支团队每天最多提交3次。

3、 排行按照得分从高到低排序，排行榜将选择团队的历史最优成绩进行排名。

## 四、作品提交要求

文件格式：预测结果文件按照csv格式提交

文件大小：无要求

提交次数限制：每支队伍每天最多3次

预测结果文件详细说明：

\1) 以csv格式提交，编码为UTF-8，第一行为表头；

\2) 提交前请确保预测结果的格式与sample_submit.csv中的格式一致。具体格式如下：

![img](https://img-blog.csdnimg.cn/img_convert/cda06c71f5ee0dcb90f845223737c52b.png)

## <u></u>

## 五、思路

简单来说，我觉得可以利用图像分类的思路来做，我们已经知道了，我们会有大概3个类别进行分类，如果用机器学习的方法的话，我们可以尝试用SVM等来进行分类

在深度学习的时代，我们利用深度学习往往能达到不错的结果，所以我尝试利用多种深度学习模型，以及在思考利用什么样的模型能得到更好的结果

### Model Coming Soon

- [x] MobileNetv2测试
- [x] DenseNet169及DenseNet201
- [x] ConvNeXt-B、ConvNeXt-L
- [x] ViT
- [x] Swim-Transformer
- [x] EfficientNetv1、v2
- [x] 集成模型的投票方法
- [x] 集成模型的均值方法
- [x] 双分类模型，先分with_mask 和 without_mask，再分 mask_weared_incorrect 和 with_mask（即是mask_weared_correct）
- [x] 增加了多个SOTA的模型进行，如Cait，Deit，Bit等

### 尝试Tricks

- [x] 尝试多用数据增强

- [x] 尝试用现有的权重进行迁移学习

- [x] 尝试利用LabelSmooth的损失

- [x] 尝试用多模型集成，模型融合等方法（

- [ ] 尝试K-Fold验证训练方法

- [ ] 尝试改变图像的分辨率，原先是224x224

- [ ] 先分类是否带了口罩，接着再进行，判断口罩佩戴是否正确。

  > 简单来说，也就是先对with_mask和without_mask进行分类，其中mask_weared_incorrect是with_mask的一部分，之后，再用另一个模型对mask_weared_incorrect和with_mask（即是mask_weared_correct）

- [x] 集成多个模型，有98%的准确率，准确率率大概提高了0.8%左右



### 部分尝试结果

- [x] 对于ConvNeXt来说，如果不是微调，也就是对于大的ConvNeXt来说，可能由于数据量太小，所以导致，我们如果对整个网络进行训练，得到的结果并不是很好，但是如果我们固定卷积层，只留下分类层进行训练，我们对此进行微调，我们得到的结果会出奇的好，训练50次以后，我们可以得到96.905%的结果。

  我也发现，ConvNeXt-B和ConvNeXt-L得到的结果是类似的。这一部分的修改可能是从模型集成，或者寻找更好的模型和Tricks进行修改。

- [x] 使用timm的库，里面含有多个预训练的模型，利用里面的Transformer模型进行训练，其中也有EfficientNetv1，v2等等

- [x] ViT效果一般，可能在少量数据集上，需要训练大量数据和大量时间才能得到更好的结果

- [x] Swin-L还是能得到很不错的结果，在一开始的就可以得到不错结果

- [x] 尝试利用数据集的均值，但是没有得到很好的结果，可能是因为，本身预训练模型的均值和方差也不是数据集的标准差和方差，所以可能结果就没有得到很好的结果。但是也可能是训练方式不同，导致结果的不同

- [x] 利用集成学习的方法并且加入LabelSmooth的训练方式，测试结果

- [x] 重写集成模型的vote和mean方法，集成多个模型进行训练，得到很好的结果，有98%的准确率，准确率率大概提高了0.8%左右

- [x] 集成模型得到很好的结果，现在有98.4%左右，排在top3

  
  

### 详细参数以及运行

**数据增强处理**

```bash
transform_train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(resize, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ])
```

batchsize 都为64，主要在预训练的神经网络上微调，这样比重新训练或者在此基础上训练能得到更好的结果，并且网络也可以更快的收敛

### 部分模型的训练

**ConvNeXt-XL**

相对于其他模型的95%和96%来说，在得分榜上可以达到97.2的准确率，所以大模型是有用的

修改了均值以后，模型准确率下降了，可能是只迭代了30次，可以尝试迭代更多的次数

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -f --cuda --net ConvNeXt-XL --num-workers 8 --epochs 50 -fe 50
```

2022.7.6日，所有模型**重新训练**，利用数据集的均值进行重新训练

因为之前的模型效果不是很好，并且利用集成模型得到的结果也不是很准确，重写方法并且重新训练进行测试

**ViT-L**

重新训练50个迭代，只进行微调

```bash
CUDA_VISIBLE_DEVICES=3 python train.py -f --cuda --net ViT-L --num-workers 4 --epochs 50 -fe 50
```

**Swin-L**

重新训练50个迭代，只进行微调

```bash
CUDA_VISIBLE_DEVICES=3 python train.py -f --cuda --net Swin-L --num-workers 8 --epochs 50 -fe 50
```

利用集成模型后，准确率率大概提高了0.8%左右

### 双分类模型

对于有些数据，比如以下数据，由于两者概率相差不大或者最大类别的概率不大于阈值

所以不能很好判断，我认为这是一个瓶颈的点，**可以细粒度分类，构建一个双分类模型进行细分**

```bash
[0.45102794 0.05648582 0.49248626] CFCZOBLXH1.jpg
[0.18592907 0.39920693 0.41486399] RLESDSXSSR.jpg
[0.30411253 0.49519148 0.20069597] NB4AN54S1Q.jpg
[0.26813637 0.33945055 0.39241309] TM034OWVES.jpg
[0.4682257  0.33433045 0.19744384] 201ZHL5JQI.jpg
[0.4337055  0.06805032 0.49824418] 3N75J18A27.jpg
[0.40534851 0.430322   0.16432948] N0B20QYZR8.jpg
[0.18397121 0.39021482 0.42581395] E5FY5X1GBM.jpg
[0.48302421 0.23825759 0.27871821] H2SGRO0KXF.jpg
[0.41494484 0.4792326  0.10582254] XSH7CLGF34.png
[0.1363276  0.43496656 0.42870582] RHF7BEZMZC.jpg
[0.40905026 0.31925034 0.27169936] J92D3JK2IQ.jpg
[0.11439361 0.4694288  0.41617759] GZXW36WEPJ.jpg
[0.31681375 0.43822954 0.2449567 ] MOJFT5KI6K.jpg
[0.34200144 0.46902096 0.1889776 ] JAZF4X8S02.jpg
[0.20950992 0.48486815 0.30562193] 0S11KWMGCD.png
```

初步划分为先分类是否带了口罩，接着再进行，判断口罩佩戴是否正确。

> 简单来说，也就是先对with_mask和without_mask进行分类，其中mask_weared_incorrect是with_mask的一部分，之后，再用另一个模型对mask_weared_incorrect和with_mask（即是mask_weared_correct）

首先需要对数据集进行分类，对原先的数据集进行更换

**mask_weared_incorrect是with_mask的一部分**  类别 with_mask without_mask 口罩佩戴细分类

```bash
cp -rf data data2
mv data2/train/mask_weared_incorrect/* data2/train/with_mask
rm -rf data2/train/mask_weared_incorrect data2/test
CUDA_VISIBLE_DEVICES=3 python train.py -f --cuda --net DenseNet161 --num-workers 8 --epochs 50 -fe 50 -nc 2 --data data2/ --checkpoint checkpoint2
rm -rf data2
```

**分类mask_weared_incorrect和with_mask**  类别 mask_weared_incorrect with_mask 口罩正确佩戴细分类

```bash
cp -rf data data3
rm -rf data3/test data3/train/without_mask
CUDA_VISIBLE_DEVICES=3 python train.py -f --cuda --net DenseNet161 --num-workers 8 --epochs 30 -fe 30 -nc 2 --data data3/ --checkpoint checkpoint3
rm -rf data3
```

2022.7.11 增加进行双分类以后，得到的结果又提高了



## 六、代码完善以及使用

对train.py进行修改，有个图像分类的代码框架，可以运用到大多数数据之中。

对predict.py进行修改，完善了整体的框架，不用一次性读入所有的模型，太占显存了，而是一次读入一个模型进行测试

<details open>
<summary>训练方式</summary>

```bash
usage: train.py [-h] [--lr LR] [--num-classes NUM_CLASSES] [--cuda] [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS]
                [--net {LeNet5,AlexNet,VGG16,VGG19,ResNet34,ResNet50,ResNet101,DenseNet,DenseNet121,DenseNet169,DenseNet201,MobileNetv1,MobileNetv2,ResNeXt50-32x4d,ResNeXt101-32x8d,EfficientNet-b0,EfficientNet-b1,EfficientNet-b2,EfficientNet-b3,EfficientNet-b4,EfficienNet-b5,EfficientNet-b6,EfficientNet-b7,EfficientNet-b8,EfficientNetv2-S,EfficientNetv2-M,EfficientNetv2-L,EfficientNetv2-XL,ConvNeXt-T,ConvNeXt-S,ConvNeXt-B,ConvNeXt-L,ConvNeXt-XL,Swin-M,Swin-L,ViT-B,ViT-L,ViT-H,CaiT-s24,CaiT-xxs24,CaiT-xxs36,DeiT-B,DeiT-T,DeiT-S,BiT-M-resnet152x4,BiT-M-resnet152x2,BiT-M-resnet101x3,BiT-M-resnet101x1}]
                [--epochs EPOCHS] [--resume] [--resume-lr RESUME_LR] [--patience PATIENCE] [--optim {sgd,adam,adamw}] [--resize RESIZE]
                [--f] [--fe FE] [--dp] [--fp16]

PyTorch Classification Training

optional arguments:
  -h, --help            show this help message and exit
  --lr LR, -lr LR       learning rate
  --num-classes NUM_CLASSES, -nc NUM_CLASSES
                        learning rate
  --cuda, -gpu          use GPU?
  --batch-size BATCH_SIZE, -bs BATCH_SIZE
                        Batch Size for Training
  --num-workers NUM_WORKERS, -nw NUM_WORKERS
                        num-workers
  --net {LeNet5,AlexNet,VGG16,VGG19,ResNet34,ResNet50,ResNet101,DenseNet,DenseNet121,DenseNet169,DenseNet201,MobileNetv1,MobileNetv2,ResNeXt50-32x4d,ResNeXt101-32x8d,EfficientNet-b0,EfficientNet-b1,EfficientNet-b2,EfficientNet-b3,EfficientNet-b4,EfficienNet-b5,EfficientNet-b6,EfficientNet-b7,EfficientNet-b8,EfficientNetv2-S,EfficientNetv2-M,EfficientNetv2-L,EfficientNetv2-XL,ConvNeXt-T,ConvNeXt-S,ConvNeXt-B,ConvNeXt-L,ConvNeXt-XL,Swin-M,Swin-L,ViT-B,ViT-L,ViT-H,CaiT-s24,CaiT-xxs24,CaiT-xxs36,DeiT-B,DeiT-T,DeiT-S,BiT-M-resnet152x4,BiT-M-resnet152x2,BiT-M-resnet101x3,BiT-M-resnet101x1}, --model {LeNet5,AlexNet,VGG16,VGG19,ResNet34,ResNet50,ResNet101,DenseNet,DenseNet121,DenseNet169,DenseNet201,MobileNetv1,MobileNetv2,ResNeXt50-32x4d,ResNeXt101-32x8d,EfficientNet-b0,EfficientNet-b1,EfficientNet-b2,EfficientNet-b3,EfficientNet-b4,EfficienNet-b5,EfficientNet-b6,EfficientNet-b7,EfficientNet-b8,EfficientNetv2-S,EfficientNetv2-M,EfficientNetv2-L,EfficientNetv2-XL,ConvNeXt-T,ConvNeXt-S,ConvNeXt-B,ConvNeXt-L,ConvNeXt-XL,Swin-M,Swin-L,ViT-B,ViT-L,ViT-H,CaiT-s24,CaiT-xxs24,CaiT-xxs36,DeiT-B,DeiT-T,DeiT-S,BiT-M-resnet152x4,BiT-M-resnet152x2,BiT-M-resnet101x3,BiT-M-resnet101x1}
                        net type
  --epochs EPOCHS, -e EPOCHS
                        Epochs
  --resume, -r          resume from checkpoint 断点续传
  --resume-lr RESUME_LR, -rlr RESUME_LR
                        断点训练时的学习率是否改变
  --patience PATIENCE, -p PATIENCE
                        patience for Early stop
  --optim {sgd,adam,adamw}, -o {sgd,adam,adamw}
                        choose optimizer
  --resize RESIZE, -rs RESIZE
                        图像的shape
  --f, -f               choose to freeze 是否使用冻结训练
  --fe FE, -fe FE       冻结训练的迭代次数
  --dp, -dp             是否使用并行训练，多GPU
  --fp16, -fp16         是否使用混合精度训练
```



```bash
CUDA_VISIBLE_DEVICES=0 python train.py -f --cuda --net NET名字 --num-workers 8 --epochs 50 -fe 50
```

</details>

<details open>
<summary>测试方式</summary>

```bash
usage: predict.py [-h]
                  [--net {LeNet5,AlexNet,VGG16,VGG19,ResNet34,ResNet50,ResNet101,DenseNet,DenseNet121,DenseNet169,DenseNet201,MobileNetv1,MobileNetv2,ResNeXt50-32x4d,ResNeXt101-32x8d,EfficientNet-b0,EfficientNet-b1,EfficientNet-b2,EfficientNet-b3,EfficientNet-b4,EfficienNet-b5,EfficientNet-b6,EfficientNet-b7,EfficientNet-b8,EfficientNetv2-S,EfficientNetv2-M,EfficientNetv2-L,EfficientNetv2-XL,ConvNeXt-T,ConvNeXt-S,ConvNeXt-B,ConvNeXt-L,ConvNeXt-XL,Swin-M,Swin-L,ViT-B,ViT-L,ViT-H,CaiT-s24,CaiT-xxs24,CaiT-xxs36,DeiT-B,DeiT-T,DeiT-S,BiT-M-resnet152x4,BiT-M-resnet152x2,BiT-M-resnet101x3,BiT-M-resnet101x1,ensemble}]
                  [--type {mean,vote}] [--num-classes NUM_CLASSES] [--threshold THRESHOLD]

PyTorch Classification Predict

optional arguments:
  -h, --help            show this help message and exit
  --net {LeNet5,AlexNet,VGG16,VGG19,ResNet34,ResNet50,ResNet101,DenseNet,DenseNet121,DenseNet169,DenseNet201,MobileNetv1,MobileNetv2,ResNeXt50-32x4d,ResNeXt101-32x8d,EfficientNet-b0,EfficientNet-b1,EfficientNet-b2,EfficientNet-b3,EfficientNet-b4,EfficienNet-b5,EfficientNet-b6,EfficientNet-b7,EfficientNet-b8,EfficientNetv2-S,EfficientNetv2-M,EfficientNetv2-L,EfficientNetv2-XL,ConvNeXt-T,ConvNeXt-S,ConvNeXt-B,ConvNeXt-L,ConvNeXt-XL,Swin-M,Swin-L,ViT-B,ViT-L,ViT-H,CaiT-s24,CaiT-xxs24,CaiT-xxs36,DeiT-B,DeiT-T,DeiT-S,BiT-M-resnet152x4,BiT-M-resnet152x2,BiT-M-resnet101x3,BiT-M-resnet101x1,ensemble}, --model {LeNet5,AlexNet,VGG16,VGG19,ResNet34,ResNet50,ResNet101,DenseNet,DenseNet121,DenseNet169,DenseNet201,MobileNetv1,MobileNetv2,ResNeXt50-32x4d,ResNeXt101-32x8d,EfficientNet-b0,EfficientNet-b1,EfficientNet-b2,EfficientNet-b3,EfficientNet-b4,EfficienNet-b5,EfficientNet-b6,EfficientNet-b7,EfficientNet-b8,EfficientNetv2-S,EfficientNetv2-M,EfficientNetv2-L,EfficientNetv2-XL,ConvNeXt-T,ConvNeXt-S,ConvNeXt-B,ConvNeXt-L,ConvNeXt-XL,Swin-M,Swin-L,ViT-B,ViT-L,ViT-H,CaiT-s24,CaiT-xxs24,CaiT-xxs36,DeiT-B,DeiT-T,DeiT-S,BiT-M-resnet152x4,BiT-M-resnet152x2,BiT-M-resnet101x3,BiT-M-resnet101x1,ensemble}, -net {LeNet5,AlexNet,VGG16,VGG19,ResNet34,ResNet50,ResNet101,DenseNet,DenseNet121,DenseNet169,DenseNet201,MobileNetv1,MobileNetv2,ResNeXt50-32x4d,ResNeXt101-32x8d,EfficientNet-b0,EfficientNet-b1,EfficientNet-b2,EfficientNet-b3,EfficientNet-b4,EfficienNet-b5,EfficientNet-b6,EfficientNet-b7,EfficientNet-b8,EfficientNetv2-S,EfficientNetv2-M,EfficientNetv2-L,EfficientNetv2-XL,ConvNeXt-T,ConvNeXt-S,ConvNeXt-B,ConvNeXt-L,ConvNeXt-XL,Swin-M,Swin-L,ViT-B,ViT-L,ViT-H,CaiT-s24,CaiT-xxs24,CaiT-xxs36,DeiT-B,DeiT-T,DeiT-S,BiT-M-resnet152x4,BiT-M-resnet152x2,BiT-M-resnet101x3,BiT-M-resnet101x1,ensemble}, -model {LeNet5,AlexNet,VGG16,VGG19,ResNet34,ResNet50,ResNet101,DenseNet,DenseNet121,DenseNet169,DenseNet201,MobileNetv1,MobileNetv2,ResNeXt50-32x4d,ResNeXt101-32x8d,EfficientNet-b0,EfficientNet-b1,EfficientNet-b2,EfficientNet-b3,EfficientNet-b4,EfficienNet-b5,EfficientNet-b6,EfficientNet-b7,EfficientNet-b8,EfficientNetv2-S,EfficientNetv2-M,EfficientNetv2-L,EfficientNetv2-XL,ConvNeXt-T,ConvNeXt-S,ConvNeXt-B,ConvNeXt-L,ConvNeXt-XL,Swin-M,Swin-L,ViT-B,ViT-L,ViT-H,CaiT-s24,CaiT-xxs24,CaiT-xxs36,DeiT-B,DeiT-T,DeiT-S,BiT-M-resnet152x4,BiT-M-resnet152x2,BiT-M-resnet101x3,BiT-M-resnet101x1,ensemble}
                        net/model type 模型的类别
  --type {mean,vote}, -t {mean,vote}
                        Ensemble type 集成模型形式
  --num-classes NUM_CLASSES, -nc NUM_CLASSES
                        分类的类别
  --threshold THRESHOLD, -td THRESHOLD
                        设置选择模型的准确率的阈值
```

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py
```

</details>

## 七、提交结果

2022.7.7，排行榜并列第一，达到准确率有0.98095的水准

2022.7.8，排行榜第一，达到准确率0.98175的评分

2022.7.10，排行榜第一，达到准确率有0.98413的评分 （1240/1260）

具体方法是利用多个模型集成，并且利用了vote的投票方式得到最后的结果。

|  ID  |   状态   |  评分   |               提交文件名                |                           提交备注                           |      提交者       |      提交时间       |
| :--: | :------: | :-----: | :-------------------------------------: | :----------------------------------------------------------: | :---------------: | :-----------------: |
|  1   | 返回分数 | 0.98413 | submit_ensemble_mean_07-10-20-09-20.csv | 利用集成模型的均值法进行预测，设置了准确率的阈值99.5，上次代码错误 | 擅长射手的pikachu | 2022-07-10 20:22:11 |
|  2   | 返回分数 | 0.57302 | submit_ensemble_mean_07-10-19-29-19.csv |       利用集成模型的均值法进行预测，设置了准确率的阈值       | 擅长射手的pikachu | 2022-07-10 19:52:11 |
|  3   | 返回分数 | 0.98413 | submit_ensemble_vote_07-09-00-00-19.csv |       利用集成模型的投票法进行预测，设置了准确率的阈值       | 擅长射手的pikachu | 2022-07-10 18:34:54 |
|  4   | 返回分数 | 0.98175 | submit_ensemble_vote_07-09-01-04-57.csv | 利用集成模型的投票法进行预测，多加了几个分类模型进行集成，使用一定阈值的准确率 | 擅长射手的pikachu | 2022-07-09 01:28:00 |
|  5   | 返回分数 | 0.98175 | submit_ensemble_vote_07-09-00-00-19.csv | 利用集成模型的投票法进行预测，多加了几个分类模型进行集成，查看最新获得的结果 | 擅长射手的pikachu | 2022-07-09 01:04:18 |
|  6   | 返回分数 | 0.98095 | submit_ensemble_mean_07-08-23-26-36.csv | 利用集成模型的均值法进行预测，多加了新的几个分类模型进行集成，查看获得的结果 | 擅长射手的pikachu | 2022-07-09 00:42:29 |
|  7   | 返回分数 | 0.98175 |          submit_ensemble2.csv           | 利用集成模型的投票法进行预测，多加了几个分类模型进行集成，查看获得的结果 | 擅长射手的pikachu | 2022-07-07 08:36:21 |
|  8   | 返回分数 | 0.98095 |           submit_Ensemble.csv           | 利用多个预训练模型进行微调，都迭代了大概50次，只修改分类层，然后利用集成模型的投票法进行预测得到 | 擅长射手的pikachu | 2022-07-06 23:16:12 |
|  9   | 返回分数 | 0.95873 |         submit_ConvNeXt-XL.csv          | 利用ImageNet1k数据集的均值以后，进行ConvNeXt的训练，总共迭代61次。 | 擅长射手的pikachu | 2022-07-05 23:43:44 |
|  10  | 返回分数 | 0.96905 |         submit_ConvNeXt-XL.csv          |  修改了数据集的均值以后，进行ConvNeXt的训练，总共迭代45次。  | 擅长射手的pikachu | 2022-07-05 12:22:09 |
|  11  | 返回分数 | 0.96508 |         submit_ConvNeXt-XL.csv          | 修改了数据集的均值以后，进行ConvNeXt的训练，总共迭代45次，得到最优结果99.96%左右 | 擅长射手的pikachu | 2022-07-05 12:18:12 |
|  12  | 返回分数 | 0.96349 |          submit_ConvNeXt-L.csv          | 试一下ConvNeXt-L的结果，固定了卷积层的数目得到的结果，只进行微调，均值方差不同了 | 擅长射手的pikachu | 2022-07-04 09:20:45 |
|  13  | 返回分数 | 0.97222 |              submit_XL.csv              | 尝试一下ConvNeXt-XL的结果，固定了卷积层的数目得到的结果，只进行微调 | 擅长射手的pikachu | 2022-07-04 09:19:26 |
|  14  | 返回分数 | 0.96032 |           submit_Ensemble.csv           | 使用多个集成模型投票，有ConvNeXt-T-B-L,DenseNet169,ViT,Swin  | 擅长射手的pikachu | 2022-07-04 09:18:33 |
|  15  | 返回分数 | 0.96905 |              submit_L.csv               | 利用ConvNeXt-L进行训练50次，固定了卷积层的数目得到的结果，只进行微调 | 擅长射手的pikachu | 2022-07-03 09:24:29 |
|  16  | 返回分数 | 0.96905 |        submit_convnext_base.csv         | 利用ConvNeXt-B进行训练50次，固定了卷积层的数目得到的结果，只进行微调 | 擅长射手的pikachu | 2022-07-03 02:25:00 |
|  17  | 返回分数 | 0.88571 |         submit_densenet201.csv          |      尝试使用DenseNet201进行训练迭代50次，得到结果测试       | 擅长射手的pikachu | 2022-07-03 02:24:36 |
|  18  | 返回分数 | 0.95317 |               submit3.csv               |   利用ConvNeXt-T进行训练20次，固定了卷积层的数目得到的结果   | 擅长射手的pikachu | 2022-07-02 23:41:35 |
|  19  | 返回分数 | 0.95952 |               submit2.csv               |               利用DenseNet169进行计算迭代50次                | 擅长射手的pikachu | 2022-07-02 21:18:10 |
|  20  | 返回分数 | 0.83016 |               submit.csv                |                   MobieNetv2进行测试的结果                   | 擅长射手的pikachu | 2022-07-02 17:51:18 |

