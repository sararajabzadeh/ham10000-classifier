import yaml
from dataclasses import dataclass


@dataclass
class Config:
    seed: int
    image_size: int
    train_test_split: float
    model_name: str
    num_classes: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    optimizer: str
    use_class_weights: bool
    
def load_config(path):
    with open(path) as f:
        data = yaml.safe_load(f)
        return Config(**data)
