from fontTools import config
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np

class Ham10000Dataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.data['dx'].unique()))}
        self.image_paths = {}
        for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            folder = self.image_dir / part
            if folder.exists():
                for p in folder.glob('*.jpg'):
                    self.image_paths[p.stem] = p
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = self.image_paths.get(row['image_id'])
        if path is None:
            raise FileNotFoundError(f"Image not found for id: {row['image_id']}")
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        label = self.class_to_idx[row['dx']]
        
        return img, int(label)
        
def build_dataloaders(config):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    chosen_train_transform = train_transform if config.use_augmentation else val_transform
    train_ds = Ham10000Dataset(config.metadata_csv, config.data_dir, transform=chosen_train_transform)
    val_ds   = Ham10000Dataset(config.metadata_csv, config.data_dir, transform=val_transform)
    
    lesion_ids = train_ds.data["lesion_id"].unique()
    train_ids, val_ids = train_test_split(
        lesion_ids,
        train_size=config.train_test_split,
        random_state=config.seed,
    )
    
    lesion_ids = train_ds.data["lesion_id"].unique()
    train_ids, val_ids = train_test_split(
        lesion_ids, train_size=config.train_test_split, random_state=config.seed
    )
    train_mask = train_ds.data["lesion_id"].isin(train_ids)
    train_idx = train_ds.data.index[train_mask].tolist()
    val_idx   = train_ds.data.index[~train_mask].tolist()

    train_loader = DataLoader(Subset(train_ds, train_idx),
                              batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(val_ds, val_idx),
                              batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, train_ds.class_to_idx


def compute_class_weights(train_loader, num_classes, device):
    subset = train_loader.dataset
    full_df = subset.dataset.data
    class_to_idx = subset.dataset.class_to_idx
    train_labels = full_df.iloc[list(subset.indices)]['dx'].map(class_to_idx).to_numpy()
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=train_labels
    )
    return torch.tensor(weights, dtype=torch.float32).to(device)