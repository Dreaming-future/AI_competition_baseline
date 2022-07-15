CUDA_VISIBLE_DEVICES=1 python train.py -f --cuda --net ConvNeXt-B --num-workers 8 --epochs 50 -fe 20
CUDA_VISIBLE_DEVICES=1 python train.py -f --cuda --net Swin-L --num-workers 8 --epochs 50 -fe 20
CUDA_VISIBLE_DEVICES=1 python train.py -f --cuda --net ViT-B --num-workers 8 --epochs 50 -fe 20