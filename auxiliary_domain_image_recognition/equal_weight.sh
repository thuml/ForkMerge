CUDA_VISIBLE_DEVICES=0 python equal_weight.py data/domainnet -d DomainNet -s c i p q r s -t c i p q r s -a resnet101 \
  --lr 0.03 --batch-size 8 --epochs 20 --seed 0 --log logs/DomainNet/equal_weight
