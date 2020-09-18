import torch
import os
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

##### default device

default_device = 0

##########################################
# SB config
# use device, LR, grad or loss, result_dir name
# run os.system 



SELECTIVE_BACKPROP = False
SB_BETA = 3
SB_HISTORY_SIZE = 1024
SB_STALNESS = 0

SB_WARMUP_EPOCH = 0
PROB_FLOOR = 0.05

UPWEIGHT_LOSS = False

# 0 = loss, 1 = grad, 2 = rand
SELECT_MODE = 0



################################
# Compaction config

CORRECT_COMPACTION = False
CONFIDENT_CORRECT_THRESH = .7

COMPACTION_STALNESS = 0
COMPACTION_SAMPLE_PROB = .1
WARMUP_EPOCH = 1


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
#torch.cuda.set_device(default_device)
device = torch.device("cuda:{}".format(default_device) if torch.cuda.is_available() else "cpu")


def _setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8889'
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def _cleanup():
    torch.distributed.destroy_process_group()
    
def maybe_init_process_group(rank, func):
    func(rank)
    
def send_data_to_device(data, rank):
    if not torch.cuda.is_available():
        return data
    return data.cuda(device)

def send_model_to_device(model, rank):
    if not torch.cuda.is_available():
        return model
    return model.cuda(device)
    #return torch.nn.DataParallel(model).cuda(device)

def exp_decay(num, iteration, decay=0):
    return num * np.exp(decay * iteration)


def exp_grow(num, iteration, rate=0):
    return num * (1 + rate)**iteration


def sigmoid_grow(iteration, rate=0):
    return 1 / (np.exp(rate * iteration) + 1)

if __name__ == '__main__':
    pass

from SB import SBSelector

SBS = SBSelector
