import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import pandas as pd
import os

def build_grayscale_vgg16(out_dim=1):
    """
    This will reconstruct the VGG16 architecture that was trained in the previous quarter
    - The .pt file that we have is saved in the location of the outputs 
    """
    model = models.vgg16(pretrained=False)

    model.features[0] = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1
    )

    # Your regression head
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(1024, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, out_dim)
    )
    return model

def load_model(model, path="~/private/fixed_vgg_pipeline/outputs/last_model.pt"):
    """
    Loads the model weights from the specified path.
    """
    # expand ~ to home directory
    path = os.path.expanduser(path)

    model.load_state_dict(
        torch.load(
            path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    )
    return model

def generate_prediction(model, input_tensor, avg, std, device='cuda' if torch.cuda.is_available() else 'cpu'): 
    """
    Docstring for generate_prediction
    
    :param model: Description
    :param df: The dataframe containing training data statistics (mean and std)
    :param input_tensor: Description
    """
    model.eval()  # Set the model to evaluation mode
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (1, C, H, W)
    
    with torch.no_grad():  # Disable gradient calculation
        input_tensor = input_tensor.to(device)
        model = model.to(device)
        output = model(input_tensor)
        output = output * std + avg  # Denormalize the output using the mean and std of the training data
    return output.cpu().numpy()
