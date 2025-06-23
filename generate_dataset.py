import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io, img_as_ubyte
import nibabel as nib

# ------------------ Utility: Natural Sort ------------------ #
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# ------------------ Stage 1: NIfTI to PNG ------------------ #
def min_max_normalization_img(nifti_img):
    array = nifti_img.get_fdata()
    return (array - array.min()) / (array.max() - array.min())

def process_subject(subject_dir, subject_id, output_dir):
    try:
        flair = nib.load(os.path.join(subject_dir, f"{subject_id}_flair.nii"))
        t1 = nib.load(os.path.join(subject_dir, f"{subject_id}_t1.nii"))
        t1ce = nib.load(os.path.join(subject_dir, f"{subject_id}_t1ce.nii"))
        t2 = nib.load(os.path.join(subject_dir, f"{subject_id}_t2.nii"))
        seg = nib.load(os.path.join(subject_dir, f"{subject_id}_seg.nii"))
    except Exception as e:
        print(f"Error loading {subject_id}: {e}")
        return

    flair_data = min_max_normalization_img(flair)
    t1_data = min_max_normalization_img(t1)
    t1ce_data = min_max_normalization_img(t1ce)
    t2_data = min_max_normalization_img(t2)
    seg_data = seg.get_fdata()

    subject_out = os.path.join(output_dir, subject_id)
    for modality in ['flair', 't1', 't1ce', 't2', 'seg']:
        os.makedirs(os.path.join(subject_out, modality), exist_ok=True)

    num_slices = flair_data.shape[2]
    for i in range(num_slices):
        io.imsave(os.path.join(subject_out, 'flair', f'{subject_id}_flair_{i}.png'), img_as_ubyte(flair_data[:, :, i]), check_contrast=False)
        io.imsave(os.path.join(subject_out, 't1', f'{subject_id}_t1_{i}.png'), img_as_ubyte(t1_data[:, :, i]), check_contrast=False)
        io.imsave(os.path.join(subject_out, 't1ce', f'{subject_id}_t1ce_{i}.png'), img_as_ubyte(t1ce_data[:, :, i]), check_contrast=False)
        io.imsave(os.path.join(subject_out, 't2', f'{subject_id}_t2_{i}.png'), img_as_ubyte(t2_data[:, :, i]), check_contrast=False)

        seg_mask = np.stack([
            seg_data[:, :, i] == 1,  # necrosis
            seg_data[:, :, i] == 4,  # enhancing
            seg_data[:, :, i] == 2   # edema
        ], axis=-1)
        io.imsave(os.path.join(subject_out, 'seg', f'{subject_id}_seg_{i}.png'), img_as_ubyte(seg_mask), check_contrast=False)

    tqdm.write(f"{subject_id} processed ({num_slices} slices).")

# ------------------ Stage 2: CSV Generation ------------------ #
def generate_csv_from_saved_images(output_dir):
    data_rows = []

    for subject_id in sorted(os.listdir(output_dir)):
        subject_path = os.path.join(output_dir, subject_id)
        flair_dir = os.path.join(subject_path, 'flair')
        seg_dir = os.path.join(subject_path, 'seg')

        if not os.path.isdir(flair_dir) or not os.path.isdir(seg_dir):
            continue

        flair_images = sorted(os.listdir(flair_dir), key=natural_sort_key)
        seg_images = sorted(os.listdir(seg_dir), key=natural_sort_key)

        for flair_file, seg_file in zip(flair_images, seg_images):
            flair_path = os.path.join(subject_path, 'flair', flair_file)
            seg_path = os.path.join(subject_path, 'seg', seg_file)

            mask = io.imread(seg_path)
            necrosis = 1 if np.any(mask[:, :, 0]) else 0
            enhancing = 1 if np.any(mask[:, :, 1]) else 0
            edema = 1 if np.any(mask[:, :, 2]) else 0
            label = 1 if (necrosis or enhancing or edema) else 0

            data_rows.append({
                'image_path': flair_path,
                'mask_path': seg_path,
                'label': label,
                'necrosis': necrosis,
                'enhancing': enhancing,
                'edema': edema
            })

    df = pd.DataFrame(data_rows)
    print(f"CSV created with {len(df)} slices.")
    return df

# ------------------ Stage 3: Split CSV ------------------ #
def split_and_save(df):
    slices_per_subject = 155
    num_train_subjects = 237
    num_val_subjects = 59

    num_train = slices_per_subject * num_train_subjects
    num_val = slices_per_subject * num_val_subjects

    train_df = df.iloc[:num_train].reset_index(drop=True)
    val_df = df.iloc[num_train:num_train + num_val].reset_index(drop=True)
    test_df = df.iloc[num_train + num_val:].reset_index(drop=True)

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print(f"\nSaved train.csv with {len(train_df)} rows")
    print(f"Saved val.csv with {len(val_df)} rows")
    print(f"Saved test.csv with {len(test_df)} rows")

# ------------------ Main: Unified Pipeline ------------------ #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified BraTS preprocessing pipeline: Convert NIfTI to PNG, generate CSV, split into train/val/test.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to input BraTS NIfTI dataset")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save 2D PNG slices")

    args = parser.parse_args()

    print(f"Converting NIfTI to PNG slices...")
    subjects = sorted(os.listdir(args.input_dir))
    for subject in tqdm(subjects, desc="Processing subjects"):
        process_subject(os.path.join(args.input_dir, subject), subject, args.output_dir)

    print("\nGenerating classification CSV...")
    df = generate_csv_from_saved_images(args.output_dir)

    print("\nSplitting into train/val/test...")
    split_and_save(df)
