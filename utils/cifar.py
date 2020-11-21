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
            self.corrputions = np.load('dataset_overrides/cifar10/75pct_corruptions.npy', allow_pickle=True).item()
            self.shuffle_idx = torch.randperm(1 * 32 * 32)
            
        
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        
        if hasattr(self, 'corrputions') and index in self.corrputions:
            
            # random label
            target = self.corrputions[index]
            
            # gaussian 
            #data = data.std() * torch.randn(data.shape) + data.mean()
            
            # permutation of pixels
            #data = torch.stack([x.view(-1)[self.shuffle_idx].view(32, 32) for x in data], dim=0)

        
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

    @staticmethod
    def classes():
        return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                'ship', 'truck')
