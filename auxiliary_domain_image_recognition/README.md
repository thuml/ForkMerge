# Auxiliary Domain Image Recognition

## Installation

It’s suggested to use **pytorch==1.10.1** and **torchvision==0.11.2** in order to reproduce our results.

After installing torch, simply run

```
pip install -r requirements.txt
```

## Dataset

Following datasets can be downloaded automatically:

- [DomainNet](http://ai.bu.edu/M3SDA/)

If you have trouble downloading from the original links provided [here](./utils/datasets/domainnet.py),
trying the links from tsinghua cloud might help (commented out in the file).

## Supported Methods

Currently, we only provide a minimal implementation, including:

- Equal Weighting (EW)
- [ForkMerge](https://arxiv.org/abs/2301.12618)

## Usage

We provide the scripts to run each method in the corresponding shell files.
For example, if you want to
train ForkMerge, use the following script

```shell script
# Train ForkMerge on DomainNet using ResNet-101.
# Choose task c as the target task and use all the other tasks including i, p, q, r, s as auxiliary tasks.
# Assume you have put the datasets under the path `data/domainnet`, 
# or you are glad to download the datasets automatically from the Internet to this path.
CUDA_VISIBLE_DEVICES=0 python forkmerge.py data/domainnet -d DomainNet -s c i p q r s \
    -t c -a resnet101 --epochs 20 -i 2500 --seed 0 \
    --log logs/DomainNet/forkmerge/c
```

## Explanation of ForkMerge Implementation

The code [here](./forkmerge.py) implements ForkMerge with pruning strategy (introduced in Section 4.2 of the paper)
to achieve better trade-off between performance and efficiency. Here we take it as an example,
and the implementations in other scenarios (scene understanding etc.) are quite similar.

1. Rank all the auxiliary tasks according to their contribution to the target task (line 227-247). 
   Concretely, for each auxiliary task, we jointly train with the target tasks for several epochs.
   Afterwards, we can sort according to the validation performance of target tasks. 
   The period of joint training is determined by hyperparameter `--pruning_epochs`. 
   Usually, a relative small value suffices.

2. Construct the general form the task weighting vector in ForkMerge (line 248-260). 
   The `--topk` hyperparameter specifies the number of auxiliary tasks to be used in each branch.
   For example, if `--topk=[0, 2]`, then there will be two branches in ForkMerge. 
   The first one is only optimized with the target task. While the second one is jointly optimized with the target task and two auxiliary tasks.
   Here, the choices of these two auxiliary tasks are determined by the ranking in step 1 (simply use the top 2).

3. Train ForkMerge (line 267-315). 
   The `--epoch_step` hyperparameter indicates the time interval of merging ($\Delta_t$ in our paper).
   And the `--alphas` hyperparameter specifies the searched interpolation weights of model parameters.
   As introduced in the Appendix, we adopt a greedy merging strategy to reduce computation cost (line 286-294).

After these three steps, we can obtain the final ForkMerge model and test its performance! 

It's also easy to reproduce our results in Table 2 of the paper. Simply set `--topk=[0, 5]`. 
In this case, all the auxiliary tasks are jointly trained in one branch.

## Citation

If you find this repository useful in your research, please cite our paper:

```
@article{jiang2023forkmerge,
   title={ForkMerge: Overcoming Negative Transfer in Multi-Task Learning},
   author={Jiang, Junguang and Chen, Baixu and Pan, Junwei and Wang, Ximei and Dapeng, Liu and Jiang, Jie and Long, Mingsheng},
   journal={arXiv preprint arXiv:2301.12618},
   year={2023}
}
```

## Contact

If you have any problem with our code, feel free to contact 

- Junguang Jiang (JiangJunguang1123@outlook.com)
- Baixu Chen (cbx_99_hasta@outlook.com)
