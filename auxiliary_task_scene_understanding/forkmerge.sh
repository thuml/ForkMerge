python forkmerge.py --weighting EW --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --source_tasks segmentation depth normal --target_tasks segmentation \
  --log logs/nyuv2/forkmerge/segmentation --epoch_step 30 --seed 0
python forkmerge.py --weighting EW --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --source_tasks segmentation depth normal --target_tasks depth \
  --log logs/nyuv2/forkmerge/depth --epoch_step 30 --seed 0
python forkmerge.py --weighting EW --arch HPS --dataset_path data/nyuv2 --gpu_id 0 --scheduler step \
  --source_tasks segmentation depth normal --target_tasks normal \
  --log logs/nyuv2/forkmerge/normal --epoch_step 30 --seed 0
