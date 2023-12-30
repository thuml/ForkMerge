CUDA_VISIBLE_DEVICES=0 python forkmerge.py data/domainnet -d DomainNet -s c i p q r s \
    -t c -a resnet101 --epochs 20 -i 2500 --seed 0 \
    --log logs/DomainNet/forkmerge/c
CUDA_VISIBLE_DEVICES=0 python forkmerge.py data/domainnet -d DomainNet -s c i p q r s \
    -t i -a resnet101 --epochs 20 -i 2500 --seed 0 \
    --log logs/DomainNet/forkmerge/i
CUDA_VISIBLE_DEVICES=0 python forkmerge.py data/domainnet -d DomainNet -s c i p q r s \
    -t p -a resnet101 --epochs 20 -i 2500 --seed 0 \
    --log logs/DomainNet/forkmerge/p
CUDA_VISIBLE_DEVICES=0 python forkmerge.py data/domainnet -d DomainNet -s c i p q r s \
    -t q -a resnet101 --epochs 20 -i 2500 --seed 0 \
    --log logs/DomainNet/forkmerge/q
CUDA_VISIBLE_DEVICES=0 python forkmerge.py data/domainnet -d DomainNet -s c i p q r s \
    -t r -a resnet101 --epochs 20 -i 2500 --seed 0 \
    --log logs/DomainNet/forkmerge/r
CUDA_VISIBLE_DEVICES=0 python forkmerge.py data/domainnet -d DomainNet -s c i p q r s \
    -t s -a resnet101 --epochs 20 -i 2500 --seed 0 \
    --log logs/DomainNet/forkmerge/s

CUDA_VISIBLE_DEVICES=0 python forkmerge.py data/domainnet -d DomainNet -s c i p q r s \
    -t c -a vit_base_patch16_224 --no-pool \
    --optimizer adamw --weight-decay 0.05 --lr 0.001 \
    --epochs 20 -i 2500 --seed 0 \
    --log logs/DomainNet/vit_base_patch16_224/forkmerge/c
CUDA_VISIBLE_DEVICES=0 python forkmerge.py data/domainnet -d DomainNet -s c i p q r s \
    -t i -a vit_base_patch16_224 --no-pool \
    --optimizer adamw --weight-decay 0.05 --lr 0.001 \
    --epochs 20 -i 2500 --seed 0 \
    --log logs/DomainNet/vit_base_patch16_224/forkmerge/i
CUDA_VISIBLE_DEVICES=0 python forkmerge.py data/domainnet -d DomainNet -s c i p q r s \
    -t p -a vit_base_patch16_224 --no-pool \
    --optimizer adamw --weight-decay 0.05 --lr 0.001 \
    --epochs 20 -i 2500 --seed 0 \
    --log logs/DomainNet/vit_base_patch16_224/forkmerge/p
CUDA_VISIBLE_DEVICES=0 python forkmerge.py data/domainnet -d DomainNet -s c i p q r s \
    -t q -a vit_base_patch16_224 --no-pool \
    --optimizer adamw --weight-decay 0.05 --lr 0.001 \
    --epochs 20 -i 2500 --seed 0 \
    --log logs/DomainNet/vit_base_patch16_224/forkmerge/q
CUDA_VISIBLE_DEVICES=0 python forkmerge.py data/domainnet -d DomainNet -s c i p q r s \
    -t r -a vit_base_patch16_224 --no-pool \
    --optimizer adamw --weight-decay 0.05 --lr 0.001 \
    --epochs 20 -i 2500 --seed 0 \
    --log logs/DomainNet/vit_base_patch16_224/forkmerge/r
CUDA_VISIBLE_DEVICES=0 python forkmerge.py data/domainnet -d DomainNet -s c i p q r s \
    -t s -a vit_base_patch16_224 --no-pool \
    --optimizer adamw --weight-decay 0.05 --lr 0.001 \
    --epochs 20 -i 2500 --seed 0 \
    --log logs/DomainNet/vit_base_patch16_224/forkmerge/s
