import argparse
import os
import pandas as pd
from torch.utils.data import DataLoader
from dataset import TrainDataset
from clnet_multi import CLNet
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train CLNet on BraTS Binary and Multiclass with Pretraining')
    
    parser.add_argument('--project_path', default='multi_cam_project')
    parser.add_argument('--record_path', default='pretrain_record')
    parser.add_argument('--modality', default='flair_t1ce_t2')
    parser.add_argument('--binary_epochs', type=int, default=100)
    parser.add_argument('--multiclass_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=155)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    parser.add_argument('--dataset_type', type=str, default='brats')

    return parser.parse_args()


def train_model(args, train_df, val_df, config, task, pretrained_path=None):
    args.task = task
    if args.task == 'binary':
        args.epochs = args.binary_epochs
    elif args.task == 'multiclass':
        args.epochs = args.multiclass_epochs
    model_dir = os.path.join(args.project_path, args.record_path, f"{task}_pretrain")
    os.makedirs(model_dir, exist_ok=True)
    args.model_name = f"{task}_pretrain"
    args.pretrained_path = pretrained_path

    train_dataset = TrainDataset(train_df, args.img_size, config, mode='train')
    val_dataset = TrainDataset(val_df, args.img_size, config, mode='train')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    clnet = CLNet(args, train_loader, val_loader)
    clnet.run()

    # return path to encoder
    encoder_name = "binary_encoder.pth" if task == "binary" else "multiclass_encoder.pth"
    return os.path.join(model_dir, encoder_name)


def main():
    args = parse_args()
    set_seed(42)

    # # --- Binary Training ---
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')

    binary_config = {
        'dataset': args.dataset_type,
        'task': 'binary',
        'combine': None
    }

    train_model(args, train_df, val_df, binary_config, task='binary')

    # --- Multiclass Training ---
    
    multiclass_config = {
        'dataset': args.dataset_type,
        'task': 'multiclass',
        'combine': {
            'core': ['necrosis', 'enhancing'],
            'edema': ['edema']
        }
    }

    train_model(args, train_df, val_df, multiclass_config, task='multiclass')


if __name__ == '__main__':
    main()
