CUDA_VISIBLE_DEVICES=2 python train.py --cuda --net Efficientv2-b0 --epochs 50 -bs 8 -lr 0.0005 -p 10 
CUDA_VISIBLE_DEVICES=2 python train.py --cuda --net Efficientv2-b1 --epochs 50 -bs 8 -lr 0.0005 -fe 10 -p 10
CUDA_VISIBLE_DEVICES=2 python train.py --cuda --net Efficientv2-b2 --epochs 50 -bs 8 -lr 0.0005 -fe 10 -p 10