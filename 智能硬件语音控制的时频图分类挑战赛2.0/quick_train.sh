CUDA_VISIBLE_DEVICES=2 python train.py -f --cuda --net ResNet18 --epochs 50 -bs 8 -lr 0.0005 -fe 5 -p 10
CUDA_VISIBLE_DEVICES=2 python train.py -f --cuda --net Efficientv2-b0 --epochs 50 -bs 8 -lr 0.0005 -fe 5 -p 10
CUDA_VISIBLE_DEVICES=2 python train.py -f --cuda --net Efficientv2-b1 --epochs 50 -bs 8 -lr 0.0005 -fe 5 -p 10
CUDA_VISIBLE_DEVICES=2 python train.py -f --cuda --net Efficientv2-b2 --epochs 50 -bs 8 -lr 0.0005 -fe 5 -p 10
CUDA_VISIBLE_DEVICES=2 python train.py -f --cuda --net Efficientv2-b3 --epochs 50 -bs 8 -lr 0.0005 -fe 5 -p 10
CUDA_VISIBLE_DEVICES=2 python train.py -f --cuda --net Efficientv2-T --epochs 50 -bs 8 -lr 0.0005 -fe 5 -p 10
CUDA_VISIBLE_DEVICES=2 python predict2.py