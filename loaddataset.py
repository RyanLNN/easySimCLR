from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy
from torchvision.transforms import ToTensor


class PreMyDataset(Dataset):

    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.ds_train = datasets.ImageFolder(root='dataset/train')
        self.classes = self.ds_train.classes

    def __getitem__(self, item):
        img, target = self.ds_train[item]
        if self.transform is not None:
            imgL = self.transform(img)
            imgR = self.transform(img)

        else:
            # 抛出异常
            raise Exception("transform is None")

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgL, imgR, target

    def __len__(self):
        return len(self.ds_train)


class PreMyDatasetStage2(Dataset):

    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.ds_train = datasets.ImageFolder(root='dataset/train')
        self.classes = self.ds_train.classes

    def __getitem__(self, item):
        img, target = self.ds_train[item]
        if self.transform is not None:
            img = self.transform(img)

        else:
            # 抛出异常
            raise Exception("transform is None")

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ds_train)

class PreDataset(CIFAR10):
    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]
        img = Image.fromarray(img)

        if self.transform is not None:
            imgL = self.transform(img)
            imgR = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgL, imgR, target


if __name__ == "__main__":
    import config

    # train_data = PreDataset(root='dataset', train=True, transform=config.train_transform, download=True)
    train_data = PreMyDataset(transform=config.my_train_transform)
    # train_data = datasets.ImageFolder(root='dataset/train')
    dataL, dataR, label = train_data[1500]
    print(dataL.shape, dataR.shape, label)
