import errno
import math
import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

__all__ = [
    'set_random_seed', 'save_pickle', 'find_most_freq', 'init_params',
    'mkdir_p', 'AverageMeter'
]


def set_random_seed(state):
    use_cuda = torch.cuda.is_available()
    manual_seed = state['manualSeed']
    # Random seed
    if manual_seed is None:
        manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(manual_seed)


def get_zero_param(model):
    total = 0
    for m in model.modules():
        if _prunable(m):
            total += torch.sum(m.weight.data.eq(0))
    return total
        
def get_conv_zero_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += torch.sum(m.weight.data.eq(0))
    return total


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def find_most_freq(a):
    (values, counts) = np.unique(a, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]


def save_pickle(name, obj):
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self, fmt=':f'):
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{val:' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
