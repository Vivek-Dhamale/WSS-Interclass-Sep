#!/bin/bash

echo "Generating dataset..."
python generate_dataset.py --input_dir ../MICCAI_BraTS2020_TrainingData --output_dir ../TrainingData_2d_images

echo "Pretraining CLNet..."
python pretrain_clnet.py --project_path "multi_cam_project" --record_path "pretrain_record" --modality "flair_t1ce_t2" --binary_epochs 100 --multiclass_epochs 50 --batch_size 155 --learning_rate 1e-3 --img_size 224 --gpu_ids 0 --dataset_type "brats"

echo "Training CNet..."
python train_cnet.py --project_path "multi_cam_project" --record_path "train_record" --modality "flair_t1ce_t2" --binary_epochs 50 --multiclass_epochs 50 --batch_size 155 --learning_rate 5e-4 --img_size 224 --gpu_ids 0 --dataset_type "brats"

echo "Training AggNet..."
python train_aggnet.py --project_path "multi_cam_project" --record_path "agg_train_record" --modality "flair_t1ce_t2" --epochs 50 --batch_size 156 --learning_rate 1e-3 --img_size 224 --gpu_ids 0 1 --dataset_type "brats"

echo "Running DesignCAM Evaluation..."
python run_design_cam.py --project_path "multi_cam_project" --record_path "agg_eval_record" --modality flair_t1ce_t2 --gpu_ids 0 --dataset_type "brats"

echo "Pipeline execution completed."
