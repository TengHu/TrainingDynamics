from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import numpy as np

__all__ = ['IndexedCifar10']

class IndexedCifar10(Dataset):
    ##### Normalization
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    
    
    def __init__(self, transform=transform_test, root='./data', train=True, download=True):
        self.cifar10 = datasets.CIFAR10(
            root=root, download=download, train=train, transform=transform)
        
        if train:
            pass
            #self.random_labels = np.load('dataset_overrides/cifar10/25pct_random_label.npy')
            #self.examples_to_add_noise = set(np.load('dataset_overrides/cifar10/25pct_example_to_add_noise.npy'))
            #self.noise = torch.randn(self.cifar10.data.shape)
        
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        
        if hasattr(self, 'random_labels'):
            target = self.random_labels[index]
            
        if hasattr(self, 'examples_to_add_noise'):
            if index in self.examples_to_add_noise:
                data = (data + self.noise[index].transpose(0,2))    
        
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

    @staticmethod
    def classes():
        return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                'ship', 'truck')
