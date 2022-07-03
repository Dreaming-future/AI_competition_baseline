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
- [ ] Swim-Transformer
- [ ] EfficientNetv1、v2

### 尝试Tricks

- [x] 尝试多用数据增强
- [x] 尝试用现有的权重进行迁移学习
- [ ] 尝试利用LabelSmooth的损失
- [ ] 尝试用多模型集成，模型融合等方法
- [ ] 尝试K-Fold验证训练方法
- [ ] 尝试改变图像的分辨率，原先是224x224
- [ ] 



### 部分尝试结果

- [x] 对于ConvNeXt来说，如果不是微调，也就是对于大的ConvNeXt来说，可能由于数据量太小，所以导致，我们如果对整个网络进行训练，得到的结果并不是很好，但是如果我们固定卷积层，只留下分类层进行训练，我们对此进行微调，我们得到的结果会出奇的好，训练50次以后，我们可以得到96.905%的结果。

  我也发现，ConvNeXt-B和ConvNeXt-L得到的结果是类似的。这一部分的修改可能是从模型集成，或者寻找更好的模型和Tricks进行修改。

- [x] 使用timm的库，里面含有多个预训练的模型，利用里面的Transformer模型进行训练，其中也有EfficientNetv1，v2等等

- [ ] ViT效果一般，可能在少量数据集上，需要训练大量数据和大量时间才能得到更好的结果

  Swin-L还是能得到很不错的结果，在一开始的就可以得到不错结果



### 详细参数以及运行

**数据增强处理**

```bash
transform_train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(resize, padding=4),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ])
```



**DenseNet169**

96.75412747640252 %

```bash
CUDA_VISIBLE_DEVICES=2 python train.py -f --cuda --net DenseNet169 --num-workers 8 --epochs 50 -fe 20
```

**ConvNeXt-T**

100%

```bash
CUDA_VISIBLE_DEVICES=2 python train.py -f --cuda --net ConvNeXt-T --num-workers 8 --epochs 50 -fe 20
```

**ConvNeXt-B**

99.734268707483 % 100%

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -f --cuda --net ConvNeXt-B --num-workers 8 --epochs 30 -fe 30
```

**ConvNeXt-L**

99.96811224489795 % 100%

```bash
CUDA_VISIBLE_DEVICES=1 python train.py -f --cuda --net ConvNeXt-L --num-workers 8 --epochs 30 -fe 30
```

**ViT-L**

运行到25epochs，显存暂时不够

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -f --cuda --net ViT-L --num-workers 8 --epochs 50 -fe 25
```

**Swin-L**

运行到25epochs，显存暂时不够

```bash
CUDA_VISIBLE_DEVICES=3 python train.py -f --cuda --net Swin-L --num-workers 8 --epochs 50 -fe 25
```



### 提交结果

|  ID  |   状态   |  评分   |        提交文件名        |                           提交备注                           |      提交者       |      提交时间       |
| :--: | :------: | :-----: | :----------------------: | :----------------------------------------------------------: | :---------------: | :-----------------: |
|  1   | 返回分数 | 0.96905 |       submit_L.csv       | 利用ConvNeXt-L进行训练50次，固定了卷积层的数目得到的结果，只进行微调 | 擅长射手的pikachu | 2022-07-03 09:24:29 |
|  2   | 返回分数 | 0.96905 | submit_convnext_base.csv | 利用ConvNeXt-B进行训练50次，固定了卷积层的数目得到的结果，只进行微调 | 擅长射手的pikachu | 2022-07-03 02:25:00 |
|  3   | 返回分数 | 0.88571 |  submit_densenet201.csv  |      尝试使用DenseNet201进行训练迭代50次，得到结果测试       | 擅长射手的pikachu | 2022-07-03 02:24:36 |
|  4   | 返回分数 | 0.95317 |       submit3.csv        |   利用ConvNeXt-T进行训练20次，固定了卷积层的数目得到的结果   | 擅长射手的pikachu | 2022-07-02 23:41:35 |
|  5   | 返回分数 | 0.95952 |       submit2.csv        |               利用DenseNet169进行计算迭代50次                | 擅长射手的pikachu | 2022-07-02 21:18:10 |
|  6   | 返回分数 | 0.83016 |        submit.csv        |                   MobieNetv2进行测试的结果                   | 擅长射手的pikachu | 2022-07-02 17:51:18 |