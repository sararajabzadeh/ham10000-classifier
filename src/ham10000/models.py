from torchvision import models
import torch.nn as nn

def build_model(num_classes):
    
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

