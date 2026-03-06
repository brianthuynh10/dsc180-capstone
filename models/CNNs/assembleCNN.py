import os
import torch
import torch.nn as nn
from torchvision import models
from config import MODEL_WEIGHTS_PATH


def load_model_weights(model, weight_filename):
    """
    Load model weights from the specified filename in MODEL_WEIGHTS_PATH.
    """
    path = os.path.join(MODEL_WEIGHTS_PATH, weight_filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def build_grayscale_cnn(out_dim=1, model_name="resnet50"):
    """
    Reconstructs the grayscale ResNet50 architecture to match the trained model checkpoint.
    """
    if model_name == "resnet50":
        # path for the ResNet weights:

        model = models.resnet50(weights=None)
        # --- Convert first conv layer to grayscale ---
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
        )
        # load in the pre-trained model weights
        model = load_model_weights(model, "resnet50_best_model.pt")

        return model

    elif model_name == "vgg16":

        model = models.vgg16(weights=None)
        model.features[0] = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
        )

        path = os.path.join(MODEL_WEIGHTS_PATH, "vgg16_last_model.pt")
        model = load_model_weights(model, "vgg16_last_model.pt")
    return model
