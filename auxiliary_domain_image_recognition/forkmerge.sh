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
