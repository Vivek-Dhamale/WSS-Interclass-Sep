import argparse
import os
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ImageDataset
from combined_aggregation_network import AggNet
from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Train AggNet on BraTS Binary and Multiclass CAMs')
    
    parser.add_argument('--project_path', default='multi_cam_project')
    parser.add_argument('--record_path', default='agg_train_record')
    parser.add_argument('--modality', default='flair_t1ce_t2')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=156)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0,1])
    parser.add_argument('--dataset_type', type=str, default='brats')
    parser.add_argument('--loss_weight', type=float, default=None)

    return parser.parse_args()

def train_model(args, train_df, val_df, config, task):
    args.task = task

    if config['task'] == 'binary':
        args.num_classes = 1
    elif config['combine'] is not None:
        args.num_classes = len(config['combine'])
    else:
        raise ValueError("For multiclass task, 'combine' must be specified in config.")

    model_dir = os.path.join(args.project_path, args.record_path, f"{task}_aggtrain")
    os.makedirs(model_dir, exist_ok=True)

    args.model_name = f"{task}_train"
    args.pretrained_path = None
    train_dataset = ImageDataset(train_df, args.img_size, config, mode='train')
    val_dataset = ImageDataset(val_df, args.img_size, config, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    aggnet = AggNet(args, train_loader, val_loader)
    aggnet.run()

    model_name = "score_model.pth"
    model_path = os.path.join(model_dir, model_name)

    return model_path

def main():
    args = parse_args()
    set_seed(42)

    # --- Binary Training ---
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')

    binary_config = {
        'dataset': args.dataset_type,
        'task': 'binary',
        'combine': None
    }
    
    args.bin_pretrained_path = os.path.join(args.project_path, 'train_record', 'binary_clstrain', 'binary_classifier.pth')

    bin_score_model_path=train_model(args, train_df, val_df, binary_config, task='binary')
    
    # --- Multiclass Training ---

    multiclass_config = {
        'dataset': args.dataset_type,
        'task': 'multiclass',
        'combine': {
            'core': ['necrosis', 'enhancing'],
            'edema': ['edema']
        }
    }

    args.multi_pretrained_path = os.path.join(args.project_path, 'train_record', 'multiclass_clstrain', 'multi_classifier.pth')
    args.bin_score_model_pretrained_path = bin_score_model_path

    train_model(args, train_df, val_df, multiclass_config, task='multiclass')


if __name__ == '__main__':
    main()
