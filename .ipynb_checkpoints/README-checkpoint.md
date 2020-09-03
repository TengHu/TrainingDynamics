
# CIFAR

pip install -r requirements.txt


python train-SB.py --dataset cifar10 --arch convnet --lr 0.1 --momentum 0.9 --train-batch 128  --test-batch 256 --save_dir ./result-6895b3ef7b0d45a4b94f44c10d5f0f49-wrn --schedule 60 120 160 --gamma 0.2 --weight-decay 0.0005 --workers 8 --epochs 200  --manualSeed 535876


python train-SB.py --dataset mnist --arch convnet --lr 0.1 --momentum 0.9 --train-batch 128  --test-batch 256 --save_dir ./result-6895b3efb0d45a4b94f44c10d5f0f49-wrn --schedule 60 120 160 --gamma 0.2 --weight-decay 0.0005 --workers 8 --epochs 200  --manualSeed 535876