from __future__ import print_function

import argparse
import os
import time
import random
import uuid
import torch.multiprocessing as mp
import torch

seeds = [535876, 161770, 291436, 260083, 490074]




def train_worker(seed):
    torch.cuda.empty_cache()
    save_dir = './result-' + str(seed)
    

    print('#' * 100, '\n')
    print('Training Model \n')
    
    
    # Cifar10
    #os.system("python train.py --dataset cifar10 --arch wrn --lr 0.1 --momentum 0.9 --train-batch 128  --test-batch 256 --save_dir ./{} --schedule 3000000 6000000 8000000 --gamma 0.2 --weight-decay 0.0005 --workers 8 --epochs 50  --manualSeed {} --selective-backprop 0 --beta 1 --upweight 0 --mode 0 --floor 0".format(save_dir, seed))
    

    # MNIST
    os.system("python train.py --dataset mnist --arch convnet --lr 0.1 --momentum 0.9 --train-batch 512  --test-batch 512 --save_dir ./{} --workers 8 --epochs 50 --schedule 2400000 3600000 --gamma 0.2 --manualSeed {} --selective-backprop 1 --beta 3 --upweight 1 --floor 0.05 --mode 0".format(save_dir, seed))


if __name__ == '__main__':
    start = time.time()
    
    try:
         mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    #for seed in seeds:
    #    print ('./result-' + str(seed))
    
    pool = mp.Pool(processes=5)
    for seed in seeds:
        pool.apply_async(train_worker, args=(seed,))
    
    pool.close()
    pool.join()
    
    print("Take {} seconds".format(time.time() - start))