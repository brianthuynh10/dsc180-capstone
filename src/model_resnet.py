# model_resnet.py
import torch
from torchvision import models
import torch.nn as nn

# Remove this line: from ResNetModel import ResNet152Regression

def make_resnet50_model():  # you can alias it as ResNet152Regression in main.py
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # regression head
    return model
