from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#from models.layers.linear import Linear

__all__ = ['fcnet']
   
class Net(nn.Module):
    def __init__(self, num_classes=10, hidden_size=8):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x.reshape(x.shape[0], -1))
        x = F.relu(x)
        
        return self.fc2(x)

def fcnet(**kwargs):
    model = Net(**kwargs)
    return model
 