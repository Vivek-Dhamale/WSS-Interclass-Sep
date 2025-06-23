import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class TrainDataset(Dataset):
    def __init__(self, dataframe, patch_size, config,mode='train'):
        """
        config: {
            'dataset': 'brats' or '__,
            'task': 'binary' or 'multiclass',
            'combine': {
                'class1': ['necrotic core', 'enhancing'],
                'class2': ['edema'],
                ...
            }
        }
        """
        self.dataframe = dataframe
        self.config = config
        self.patch_size = patch_size

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor()
            ])
        else:
            # For validation or testing, use center crop and no augmentation
            self.transform = transforms.Compose([
                transforms.CenterCrop(patch_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.dataframe)

    def _load_image(self, index):
        if self.config["dataset"] == "brats":
            img_name_flair = self.dataframe.iloc[index, 0]
            img_name_t1ce = img_name_flair.replace("flair", "t1ce")
            img_name_t2 = img_name_flair.replace("flair", "t2")

            image_flair = Image.open(img_name_flair).convert("L")
            image_t1ce = Image.open(img_name_t1ce).convert("L")
            image_t2 = Image.open(img_name_t2).convert("L")

            image = Image.merge("RGB", (image_flair, image_t1ce, image_t2))

        return image

    def _process_labels(self, index):
        if self.config["task"] == "binary":
            return torch.tensor(self.dataframe.iloc[index, 2], dtype=torch.long).unsqueeze(0) # direct binary label
        else:
            label_dict = self.dataframe.iloc[index, 3:].to_dict()  # after 'label' column
            combined_labels = {}
            for new_class, old_classes in self.config["combine"].items():
                combined_labels[new_class] = int(any(label_dict.get(cls, 0) for cls in old_classes))

            keys = sorted(self.config['combine'].keys())
            return torch.tensor([combined_labels[k] for k in keys], dtype=torch.long)


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = self._load_image(index)
        label = self._process_labels(index)

        pos_1 = self.transform(image)
        pos_2 = self.transform(image)

        return pos_1, pos_2, label


class ImageDataset(Dataset):
    def __init__(self, dataframe, patch_size, config, mode='train'):
        self.dataframe = dataframe
        self.config = config
        self.mode = mode

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(90),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(patch_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.dataframe)

    def _load_image(self, index):
        if self.config["dataset"] == "brats":
            img_name_flair = self.dataframe.iloc[index, 0]
            img_name_t1ce = img_name_flair.replace("flair", "t1ce")
            img_name_t2 = img_name_flair.replace("flair", "t2")

            image_flair = Image.open(img_name_flair).convert("L")
            image_t1ce = Image.open(img_name_t1ce).convert("L")
            image_t2 = Image.open(img_name_t2).convert("L")

            image = Image.merge("RGB", (image_flair, image_t1ce, image_t2))

        return image

    def _process_labels(self, index):
        if self.config["task"] == "binary":
            return torch.tensor(self.dataframe.iloc[index, 2],dtype=torch.long).unsqueeze(0)  # direct binary label
        else:
            label_dict = self.dataframe.iloc[index, 3:].to_dict()
            combined_labels = {}
            for new_class, old_classes in self.config["combine"].items():
                combined_labels[new_class] = int(any(label_dict.get(cls, 0) for cls in old_classes))

            
            keys = sorted(self.config['combine'].keys())
            return torch.tensor([combined_labels[k] for k in keys], dtype=torch.long)


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = self._load_image(index)
        label = self._process_labels(index)
        transformed = self.transform(image)

        return transformed, label
    

class InferenceDataset(Dataset):
    def __init__(self, dataframe, patch_size, config):
        self.dataframe = dataframe
        self.config = config

    
        self.transform = transforms.Compose([
                transforms.CenterCrop(patch_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.dataframe)

    def _load_image(self, index):
        if self.config["dataset"] == "brats":
            img_name= self.dataframe.iloc[index, 0]
            img_name=img_name.split('/')[-1]
            img_name_flair = self.dataframe.iloc[index, 0]
            img_name_t1ce = img_name_flair.replace("flair", "t1ce")
            img_name_t2 = img_name_flair.replace("flair", "t2")

            image_flair = Image.open(img_name_flair).convert("L")
            image_t1ce = Image.open(img_name_t1ce).convert("L")
            image_t2 = Image.open(img_name_t2).convert("L")
            img_name_seg = img_name_flair.replace('flair', 'seg')

            image = Image.merge("RGB", (image_flair, image_t1ce, image_t2))
            seg = Image.open(img_name_seg).convert("RGB")

        return img_name, image, seg

    def _process_labels(self, index):
        if self.config["task"] == "binary":
            return torch.tensor(self.dataframe.iloc[index, 2],dtype=torch.long).unsqueeze(0)  # direct binary label
        else:
            label_dict = self.dataframe.iloc[index, 3:].to_dict()
            combined_labels = {}
            for new_class, old_classes in self.config["combine"].items():
                combined_labels[new_class] = int(any(label_dict.get(cls, 0) for cls in old_classes))

            
            keys = sorted(self.config['combine'].keys())
            return torch.tensor([combined_labels[k] for k in keys], dtype=torch.long)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name, image, seg = self._load_image(index)
        transformed = self.transform(image)
        seg = self.transform(seg)

        return img_name, transformed, seg