import argparse
import os
import pandas as pd
from torch.utils.data import DataLoader
from dataset import InferenceDataset
from design_cam import Design_CAM
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train AggNet on BraTS Binary and Multiclass CAMs')
    
    parser.add_argument('--project_path', default='multi_cam_project')
    parser.add_argument('--record_path', default='eval_record_design_cam')
    parser.add_argument('--modality', default='flair_t1ce_t2')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    parser.add_argument('--dataset_type', type=str, default='brats')
    
    return parser.parse_args()

def eval_model(args, train_df, val_df, test_df, config, task):
    args.task = task

    if config['task'] == 'binary':
        args.num_classes = 1
    elif config['combine'] is not None:
        args.num_classes = len(config['combine'])
    else:
        raise ValueError("For multiclass task, 'combine' must be specified in config.")
    

    model_dir = os.path.join(args.project_path, args.record_path, f"{task}_eval")
    os.makedirs(model_dir, exist_ok=True)

    args.model_name = f"{task}_train"
    args.pretrained_path = None
    train_dataset = InferenceDataset(train_df, args.img_size, config)
    val_dataset = InferenceDataset(val_df, args.img_size, config)
    test_dataset = InferenceDataset(test_df, args.img_size, config)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)   

    des_cam = Design_CAM(args)
    des_cam.run_tumor_test(test_loader, threshold=0.5)
    del des_cam

def main():
    args = parse_args()
    set_seed(42)

    # --- Multiclass inference ---
    args.batch_size = 1
    multi_train_df = pd.read_csv('train.csv')
    multi_val_df = pd.read_csv('val.csv')
    multi_test_df = pd.read_csv('test.csv')
    args.img_size = 224

    multiclass_config = {
        'dataset': args.dataset_type,
        'task': 'multiclass',
        'combine': {
            'core': ['necrosis', 'enhancing'],
            'edema': ['edema']
        }
    }

    args.bin_pretrained_path = os.path.join(args.project_path, 'train_record', 'binary_clstrain', 'binary_classifier.pth')
    args.multi_pretrained_path = os.path.join(args.project_path, 'train_record', 'multiclass_clstrain', 'multi_classifier.pth')
    args.bin_score_model_pretrained_path = os.path.join(args.project_path, 'agg_train_record', 'binary_aggtrain', 'score_model.pth')
    args.multi_score_model_pretrained_path = os.path.join(args.project_path, 'agg_train_record', 'multiclass_aggtrain', 'score_model.pth')

    eval_model(args, multi_train_df, multi_val_df, multi_test_df, multiclass_config, task='multiclass')

if __name__ == '__main__':
    main()
