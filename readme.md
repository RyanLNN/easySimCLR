# 介绍
SimCLR(A Simple Framework for Contrastive Learning of Visual Representations)是一种对比学习网络，可以对含有少量标签的数据集进行训练推理,它包含无监督学习和有监督学习两个部分。

无监督学习网络特征提取采用resnet50，将输入层进行更改，并去掉池化层及全连接层。之后将特征图平坦化，并依次进行全连接、批次标准化、relu激活、全连接，得到输出特征。

有监督学习网络使用无监督学习网络的特征提取层及参数，之后由一个全连接层得到分类输出。

## 运行 showbyvisdom.py
先执行visdom
```bash
python -m visdom.server
```

再执行showbyvisdom.py
```bash
python showbyvisdom.py
```

## 运行程序
训练过程
```bash
python trainstage1.py  # 运行上游任务

# 修改config.py中的pre_model路径，选择loss最小的那个。
python trainstage2.py --batch_size 128
```

eval过程，注意加载eval_dataset的时候，需要选择何种transform方法
```bash
# 修改config.py中的pre_model_state2的路径
python eval.py
```
