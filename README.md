This project is working in progress

pip install -r requirements.txt


* LR Scheduler is based on number of backpropagations
* Look at IndexedMNIST, IndexedCifar10 for label randomization
* Switches for selective backprops are in train_config.py


# Run Kath on CIFAR10 
python train.py --dataset cifar10 --arch wrn --lr 0.1 --momentum 0.9 --train-batch 128  --test-batch 256 --save_dir ./deleteme --schedule 3000000 6000000 8000000 --gamma 0.2 --weight-decay 0.0005 --workers 8 --epochs 30  --manualSeed 535876  --kath 1 --kath-pool 1024

python train.py --dataset mnist --arch convnet --lr 0.1 --momentum 0.9 --train-batch 512  --test-batch 512 --save_dir ./delete --workers 8 --epochs 2 --schedule 2400000 3600000 --gamma 0.2 --manualSeed 535876 --kath 1 --kath-pool 1024 --saveModel 1

# Run SB on CIFAR10 
python train.py --dataset cifar100 --arch wrn --lr 0.1 --momentum 0.9 --train-batch 128  --test-batch 256 --save_dir ./result-deleteme --schedule 3000000 6000000 8000000 --gamma 0.2 --weight-decay 0.0005 --workers 8 --epochs 5000  --manualSeed 535876 --selective-backprop 0 --beta 1 --upweight 0 --mode 0 --floor 0

python train.py --dataset cifar10 --arch wrn --lr 0.1 --momentum 0.9 --train-batch 128  --test-batch 256 --save_dir ./result-beta3-loss --schedule 3000000 6000000 8000000 --gamma 0.2 --weight-decay 0.0005 --workers 8 --epochs 10  --manualSeed 535876 --selective-backprop 1 --beta 4 --upweight 0 --mode 3 --floor 0


# MNIST
python train.py --dataset mnist --arch convnet --lr 0.1 --momentum 0.9 --train-batch 512  --test-batch 512 --save_dir ./delete --workers 8 --epochs 5 --schedule 2400000 3600000 --gamma 0.2 --manualSeed 535876 --selective-backprop 0 --beta 4 --upweight 0 --floor 0 --mode 3 --saveModel 0

