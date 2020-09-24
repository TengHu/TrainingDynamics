import torch
import os
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

##### default device
#default_device = 0

# noisy type, level in dataset
# shape of distribution in BatchedRelativeProbabilityCalculator._calculate_probability


VALID_SIZE = 5000
torch.cuda.set_device(0)

##########################################

LOSS_TYPE = 'mean'

##########################################

LOG_TO_DISK = True

##########################################
# Device Management
##########################################

# for multiprocessing only
WORLD_SIZE = 2 

# caused the imbalance memory on both gpus


#device = torch.device("cuda:{}".format(default_device) if torch.cuda.is_available() else "cpu")


def _setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def _cleanup():
    torch.distributed.destroy_process_group()
    
def maybe_init_process_group(rank, func):
    func(rank)
    
def send_data_to_device(data, rank):
    if not torch.cuda.is_available():
        return data
    return data.cuda(rank)

def send_model_to_device(model, rank):
    if not torch.cuda.is_available():
        return model
    return model.cuda(rank)
    #return torch.nn.DataParallel(model).cuda(device)

def exp_decay(num, iteration, decay=0):
    return num * np.exp(decay * iteration)


def exp_grow(num, iteration, rate=0):
    return num * (1 + rate)**iteration


def sigmoid_grow(iteration, rate=0):
    return 1 / (np.exp(rate * iteration) + 1)

if __name__ == '__main__':
    pass


