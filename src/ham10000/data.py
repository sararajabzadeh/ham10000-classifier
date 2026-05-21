from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from pathlib import Path


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
        
        return img, label
        
    