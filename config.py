import os
from torchvision import transforms
from torchvision import datasets
import numpy

use_gpu = True
gpu_name = 1

pre_model = os.path.join('pth', 'model_stage1_epoch100.pth')
pre_model_state2 = os.path.join('pth', 'model_stage2_epoch200.pth')  # 这个地方不太确定

save_path = "pth"


# 计算自己数据集的均值和方差

def compute_my_mead_std():
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((50, 50), antialias=False),
        transforms.ToTensor(),
    ])
    my_dataset = datasets.ImageFolder(root='dataset/train', transform=transform_train)
    data_r = numpy.dstack([my_dataset[i][0][:, :, 0] for i in range(len(my_dataset))])
    data_g = numpy.dstack([my_dataset[i][0][:, :, 1] for i in range(len(my_dataset))])
    data_b = numpy.dstack([my_dataset[i][0][:, :, 2] for i in range(len(my_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)
    return mean, std


my_mean, my_std = compute_my_mead_std()
# my_mean, my_std = [0.4857567, 0.485713, 0.48441252], [0.3228155, 0.32101303, 0.32024652]
# print(my_mean, my_std)

my_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(50),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # 以一定概率对图像的亮度、对比度、饱和度和色相进行变化
    transforms.RandomGrayscale(p=0.2),  # 以一定概率对图像进行灰度变换
    transforms.Normalize(my_mean, my_std)])

my_test_transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize(my_mean, my_std)])

# 以下是cifar10的均值和方差
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
