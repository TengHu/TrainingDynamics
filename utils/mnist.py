from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.utils import murmurhash3_32
import random
import torch
import numpy as np

__all__ = ['IndexedMNIST']


class IndexedMNIST(Dataset):
    """https://github.com/pytorch/examples/blob/master/mnist/main.py"""
    MEAN = 0.1307
    STD = 0.3081
    _transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((MEAN,), (STD,))])

    transform_train = _transform
    transform_test = _transform
    
    
    
    def __init__(self, transform=_transform, root='./data', train=True, download=True):
        self.mnist = datasets.MNIST(
            root=root, download=download, train=train, transform=transform)
        
        if train:
            pass
            #self.random_labels = np.load('dataset_overrides/mnist/75pct_random_label.npy')
            #self.examples_to_add_noise = set(np.load('dataset_overrides/mnist/50pct_example_to_add_noise.npy'))
            #self.noise = torch.randn(self.mnist.data.shape)
                
        self.train = train

    def __getitem__(self, index):
        data, target = self.mnist[index]
        
        
        if hasattr(self, 'random_labels'):
            target = self.random_labels[index]
            
        if hasattr(self, 'examples_to_add_noise'):
            if index in self.examples_to_add_noise:
                data = (data + self.noise[index].unsqueeze(0))
        
        return data, target, index

    def __len__(self):
        return len(self.mnist)

    @staticmethod
    def classes():
        return (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

