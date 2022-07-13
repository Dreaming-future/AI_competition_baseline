rm -rf data2
cp -r data data2
mv data2/train/mask_weared_incorrect/* data2/train/with_mask
rm -rf data2/train/mask_weared_incorrect data2/test
CUDA_VISIBLE_DEVICES=0 python train.py -f --cuda --net ResNet50 --num-workers 16 --epochs 100 -fe 30 -nc 2 -p 100 --data data2/ --checkpoint checkpoint2 -lr 0.001
rm -rf data2

rm -rf data3
cp -rf data data3
rm -rf data3/test data3/train/without_mask
CUDA_VISIBLE_DEVICES=0 python train.py -f --cuda --net ResNet50 --num-workers 16 --epochs 100 -fe 30 -nc 2 -p 100 --data data3/ --checkpoint checkpoint3 -lr 0.001
rm -rf data3