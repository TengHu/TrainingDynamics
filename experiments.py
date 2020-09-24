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
    id = uuid.uuid4().hex
    save_dir = './result-' + '-'.join((seed))
    

    print('#' * 100, '\n')
    print('Training Model\n')
    

    # Train Model, log the training examples
    os.system("python train.py --dataset mnist --arch convnet --lr 0.1 --momentum 0.9 --train-batch 512  --test-batch 512 --save_dir ./result-deleteme1 --workers 8 --epochs 50 --schedule 2400000 3600000 --gamma 0.2 --manualSeed {} --selective-backprop 1 --beta 3 --upweight 1 --floor 0.05 --mode 0".format(seed))


if __name__ == '__main__':
    start = time.time()
    pool = mp.Pool(processes=5)
    for seed in seeds:
        print (seed)
        pool.apply_async(train_worker, (seed, ))
    
    pool.close()
    pool.join()
    print("Take {} seconds".format(time.time() - start))