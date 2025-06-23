# Inter-class Separability Loss for Weakly Supervised Mutually Exclusive Multiclass Segmentation of Brain Tumor Lesions [MICCAI 2025]

This repository contains the official PyTorch implementation for our MICCAI 2025 paper:

> **Inter-class Separability Loss for Weakly Supervised Mutually Exclusive Multiclass Segmentation of Brain Tumor Lesions**  
> Vivek Dhamale, Vaanathi Sundaresan

## 📁 Dataset

We use the [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) dataset for 2D slice-level tumor classification and segmentation.

Preprocess BraTS NIfTI scans into 2D PNG slices:
```bash
python generate_dataset.py \
  --input_dir <path to your BraTS NIfTI scans folder> \
  --output_dir <path to your output dir>
```

This will also generate `train.csv`, `val.csv`, and `test.csv`.

### 📁 Folder Structure

After running the generate_dataset script, the following folder structure will be created inside your specified `--output_dir`:

```
<output_dir>/
├── <subject_id_1>/
│   ├── flair/
│   │   ├── <subject_id_1>_flair_0.png
│   │   ├── <subject_id_1>_flair_1.png
│   │   └── ...
│   ├── t1/
│   │   └── <subject_id_1>_t1_*.png
│   ├── t1ce/
│   │   └── <subject_id_1>_t1ce_*.png
│   ├── t2/
│   │   └── <subject_id_1>_t2_*.png
│   └── seg/
│       └── <subject_id_1>_seg_*.png
├── <subject_id_2>/
│   └── ...
└── ...
```


## 🚀 Training Pipeline

We follow a 3-stage training pipeline:

### 1. Pre-train CAM networks with contrastive learning

```bash
python pretrain_clnet.py \
  --project_path "multi_cam_project" \
  --record_path "pretrain_record" \
  --modality "flair_t1ce_t2" \
  --binary_epochs 100 --multiclass_epochs 50 \
  --batch_size 155 --learning_rate 1e-3 \
  --img_size 224 --gpu_ids 0 \
  --dataset_type "brats"
```

### 2. Train multilabel classification network (CNet)

```bash
python train_cnet.py \
  --project_path "multi_cam_project" \
  --record_path "train_record" \
  --modality "flair_t1ce_t2" \
  --binary_epochs 50 --multiclass_epochs 50 \
  --batch_size 155 --learning_rate 5e-4 \
  --img_size 224 --gpu_ids 0 \
  --dataset_type "brats"
```

### 3. Train aggregation network (AggNet)

```bash
python train_aggnet.py \
  --project_path "multi_cam_project" \
  --record_path "agg_train_record" \
  --modality "flair_t1ce_t2" \
  --epochs 50 --batch_size 156 \
  --learning_rate 1e-3 --img_size 224 \
  --gpu_ids 0 1 --dataset_type "brats"
```

## 📊 Evaluation

Run design-CAM segmentation and evaluation:

```bash
python run_design_cam.py \
  --project_path "multi_cam_project" \
  --record_path "agg_eval_record" \
  --modality flair_t1ce_t2 \
  --gpu_ids 0 \
  --dataset_type "brats"
```
