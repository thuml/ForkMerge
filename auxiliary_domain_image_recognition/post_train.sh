# Assume you have obtained the model trained with equal weight.
# You can also change the path to the model by modifying the --pretrained argument.
CUDA_VISIBLE_DEVICES=0 python equal_weight.py data/domainnet -d DomainNet -s c -t c -a resnet101 \
  --pretrained logs/DomainNet/equal_weight/checkpoints/best.pth \
  --lr 0.03 --batch-size 8 --epochs 20 --seed 0 --log logs/DomainNet/post_train/c
CUDA_VISIBLE_DEVICES=0 python equal_weight.py data/domainnet -d DomainNet -s i -t i -a resnet101 \
  --pretrained logs/DomainNet/equal_weight/checkpoints/best.pth \
  --lr 0.03 --batch-size 8 --epochs 20 --seed 0 --log logs/DomainNet/post_train/i
CUDA_VISIBLE_DEVICES=0 python equal_weight.py data/domainnet -d DomainNet -s p -t p -a resnet101 \
  --pretrained logs/DomainNet/equal_weight/checkpoints/best.pth \
  --lr 0.03 --batch-size 8 --epochs 20 --seed 0 --log logs/DomainNet/post_train/p
CUDA_VISIBLE_DEVICES=0 python equal_weight.py data/domainnet -d DomainNet -s q -t q -a resnet101 \
  --pretrained logs/DomainNet/equal_weight/checkpoints/best.pth \
  --lr 0.03 --batch-size 8 --epochs 20 --seed 0 --log logs/DomainNet/post_train/q
CUDA_VISIBLE_DEVICES=0 python equal_weight.py data/domainnet -d DomainNet -s r -t r -a resnet101 \
  --pretrained logs/DomainNet/equal_weight/checkpoints/best.pth \
  --lr 0.03 --batch-size 8 --epochs 20 --seed 0 --log logs/DomainNet/post_train/r
CUDA_VISIBLE_DEVICES=0 python equal_weight.py data/domainnet -d DomainNet -s s -t s -a resnet101 \
  --pretrained logs/DomainNet/equal_weight/checkpoints/best.pth \
  --lr 0.03 --batch-size 8 --epochs 20 --seed 0 --log logs/DomainNet/post_train/s
