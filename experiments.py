from __future__ import print_function

import argparse
import os
import time
import random
import uuid
import torch.multiprocessing as mp
import torch

seeds = [535876, 161770, 291436, 260083, 490074]




def train_worker(rank, seed):
    torch.cuda.empty_cache()
    save_dir = './result-' + str(seed)
    

    print('#' * 100, '\n')
    print('Training Model \n')
    
    
    # Cifar10
    os.system("python train.py --dataset cifar10 --arch wrn --lr 0.1 --momentum 0.9 --train-batch 128  --test-batch 256 --save_dir ./{} --schedule 60 120 160 --gamma 0.2 --weight-decay 0.0005 --workers 8 --epochs 200  --manualSeed {} --selective-backprop 0 --beta 1 --upweight 0 --mode 0 --floor 0.05 --rank {}".format(save_dir, seed, rank))
    

    # MNIST
    #os.system("python train.py --dataset mnist --arch convnet --lr 0.1 --momentum 0.9 --train-batch 512  --test-batch 512 --save_dir ./{} --workers 8 --epochs 50 --schedule 2400000 3600000 --gamma 0.2 --manualSeed {} --selective-backprop 1 --beta 3 --upweight 1 --floor 0.05 --mode 0 --rank {}".format(save_dir, seed, rank))


if __name__ == '__main__':
    start = time.time()
    
    try:
         mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    #for seed in seeds:
    #    print ('./result-' + str(seed))
    
    pool = mp.Pool(processes=5)
    for i, seed in enumerate(seeds):
        pool.apply_async(train_worker, args=(i, seed,))
    
    pool.close()
    pool.join()
    
    print("Take {} seconds".format(time.time() - start))