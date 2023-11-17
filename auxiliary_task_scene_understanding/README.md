# Auxiliary Task Scene Understanding

## Installation

Our code is based on [LibMTL](https://github.com/median-research-group/LibMTL). For convenience, we copy the code of LibMTL into this repository. 
And there are some minor modifications to the original code. 

It’s suggested to use **pytorch==1.10.1** and **torchvision==0.11.2** in order to reproduce our results.

After installing torch, simply run

```
pip install -r requirements.txt
```

## Dataset

Currently, only NYUv2 dataset is supported. 
To use it, you should manually download the dataset from the [Official Link](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). 
You can also download it from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/6d0a89f4ca1347d8af5f/?dl=1).

After downloading the dataset, you should put it under the path `data/nyuv2`. The structure of the dataset should be like

```
data/nyuv2
├── train
│   ├── depth
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   ├── ...
│   ├── ...
├── val
```

## Supported Methods

This codebase supports all methods already implemented in LibMTL, including

- Equal Weighting (EW)
- [GradNorm](https://proceedings.mlr.press/v80/chen18a/chen18a.pdf)
- [Uncertainty Weighting (UW)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)
- [MGDA](https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)
- [Dynamic Weight Average (DWA)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)
- [Geometric Loss Strategy (GLS)](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf)
- [Gradient Surgery (PCGrad)](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
- [GradDrop](https://papers.nips.cc/paper/2020/hash/16002f7a455a94aa4e91cc34ebdb9f2d-Abstract.html)
- [Impartial Multi-Task Learning (IMTL)](https://openreview.net/forum?id=IMPnRXEWpvr)
- [Gradient Vaccine (GradVac)](https://openreview.net/forum?id=F1vEjWK-lH_)
- [Conflict-Averse Gradient Descent (CAGrad)](https://openreview.net/forum?id=_61Qh8tULj_)
- [Nash-MTL](https://proceedings.mlr.press/v162/navon22a/navon22a.pdf)

Besides, we provide implementations of the following methods

- [GCS](https://arxiv.org/abs/1812.02224)
- [OL-AUX](https://papers.nips.cc/paper_files/paper/2019/hash/0e900ad84f63618452210ab8baae0218-Abstract.html)
- [ARML](https://arxiv.org/abs/2010.08244)
- [ForkMerge](https://arxiv.org/abs/2301.12618)

## Usage

We provide the scripts to run each baseline [train_nyu.sh](train_nyu.sh). 
The script to run ForkMerge is in [forkmerge.sh](forkmerge.sh).
For example, if you want to train ForkMerge, use the following script

```shell script
# Train ForkMerge on NYUv2, where segmentation task is chosen as the target task.
python forkmerge.py --weighting EW --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --source_tasks segmentation depth normal --target_tasks segmentation \
  --log logs/nyuv2/forkmerge/segementation --epoch_step 30 --seed 0
```

#### Hyperparameters of ForkMerge
- --epoch_step: the time interval of merging ($\Delta_t$ in our paper).
- --pruning_epochs: the period of training to rank each auxiliary task for later pruning.
- --topk: the number of auxiliary tasks to be used in each branch.
- --alphas: the searched interpolation weights of model parameters.

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

## Acknowledgments

We appreciate the following github repos for their valuable codebase:

- https://github.com/median-research-group/LibMTL
- https://github.com/lorenmt/mtan
