# equal weight
python train_nyu.py --weighting EW --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --tasks segmentation depth normal --target_tasks segmentation depth normal \
  --log logs/nyuv2/equal_weight --seed 0
# gradnorm
python train_nyu.py --weighting GradNorm --arch HPS --dataset_path data/nyuv2 --rep_grad --gpu_id 0 --scheduler step \
  --log logs/nyuv2/gradnorm --seed 0
# uncertainty
python train_nyu.py --weighting UW --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --log logs/nyuv2/uncertainty --seed 0
# mgda
python train_nyu.py --weighting MGDA --arch HPS --dataset_path data/nyuv2 --rep_grad --gpu_id 0 --scheduler step \
  --log logs/nyuv2/mgda --seed 0
# dwa
python train_nyu.py --weighting DWA --arch HPS --dataset_path data/nyuv2 --rep_grad --gpu_id 0 --scheduler step \
  --log logs/nyuv2/dwa --seed 0
# pcgrad
python train_nyu.py --weighting PCGrad --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --log logs/nyuv2/pcgrad --seed 0
# cagrad
python train_nyu.py --weighting CAGrad --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --log logs/nyuv2/cagrad --seed 0
# imtl
python train_nyu.py --weighting IMTL --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --log logs/nyuv2/imtl --seed 0
# nash_mtl
python train_nyu.py --weighting Nash_MTL --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --log logs/nyuv2/nash_mtl --seed 0
# gradvac
python train_nyu.py --weighting GradVac --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --log logs/nyuv2/gradvac --seed 0
# gcs
python train_nyu.py --weighting GCS --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --pri_tasks 1 0 0 --target_tasks segmentation --log logs/nyuv2/gcs/segmentation --seed 0
python train_nyu.py --weighting GCS --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --pri_tasks 0 1 0 --target_tasks depth --log logs/nyuv2/gcs/depth --seed 0
python train_nyu.py --weighting GCS --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --pri_tasks 0 0 1 --target_tasks normal --log logs/nyuv2/gcs/normal --seed 0
# ol_aux
python train_nyu.py --weighting OLAUX --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --pri_idx 0 --target_tasks segmentation \
  --beta 1e-4 --log logs/nyuv2/ol_aux/segmentation --seed 0
python train_nyu.py --weighting OLAUX --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --pri_idx 1 --target_tasks depth \
  --beta 1e-4 --log logs/nyuv2/ol_aux/depth --seed 0
python train_nyu.py --weighting OLAUX --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --pri_idx 2 --target_tasks normal \
  --beta 1e-4 --log logs/nyuv2/ol_aux/normal --seed 0
# arml
python train_nyu.py --weighting ARML --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --pri_idx 0 --target_tasks segmentation \
  --log logs/nyuv2/arml/segmentation --seed 0
python train_nyu.py --weighting ARML --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --pri_idx 1 --target_tasks depth \
  --log logs/nyuv2/arml/depth --seed 0
python train_nyu.py --weighting ARML --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --pri_idx 2 --target_tasks normal \
  --log logs/nyuv2/arml/normal --seed 0
