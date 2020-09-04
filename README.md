Working in Progress !!


pip install -r requirements.txt

# CIFAR10
python train.py --dataset cifar10 --arch convnet --lr 0.1 --momentum 0.9 --train-batch 128  --test-batch 256 --save_dir ./result-689f7b0d45a4b94f44c10d5f0f49-wrn --schedule 60 120 160 --gamma 0.2 --weight-decay 0.0005 --workers 8 --epochs 200  --manualSeed 535876


# MNIST
python train.py --dataset mnist --arch convnet --lr 0.1 --momentum 0.9 --train-batch 128  --test-batch 256 --save_dir ./result-6895b3efb0d45a4b94f44c10d5f0f49-wrn --schedule 60 120 160 --gamma 0.2 --weight-decay 0.0005 --workers 8 --epochs 200  --manualSeed 535876


Switches for selective backprops are in train_config.py