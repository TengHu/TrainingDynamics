from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.utils import murmurhash3_32
import random
import torch
import numpy as np
import torchvision.transforms as transforms

__all__ = ['IndexedFashion']


FULL_TRAINSET_EVALUATE_BATCH_SIZE = 600

class IndexedFashion(Dataset):
    
    MEAN = 0.1307
    STD = 0.3081
    _transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((MEAN,), (STD,))])
    transform_train = _transform
    transform_test = _transform
    
    
    def __init__(self, transform=_transform, root='./data', train=True, download=True):
        self.fashion = datasets.FashionMNIST(
            root=root, download=download, train=train, transform=transform)
        self.train = train

    def __getitem__(self, index):
        data, target = self.fashion[index]
        return data, target, index

    def __len__(self):
        return len(self.fashion)

    @staticmethod
    def classes():
        return ('T-shirt/top',
                 'Trouser',
                 'Pullover',
                 'Dress',
                 'Coat',
                 'Sandal',
                 'Shirt',
                 'Sneaker',
                 'Bag',
                 'Ankle boot')

