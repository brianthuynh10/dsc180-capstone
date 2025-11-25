import torch.nn as nn
from torchvision import models

def make_vgg_model(out_dim=1, pretrained=True, model_depth=16):
    """
    Create a VGG-based regression model for grayscale X-ray input.
    Works with vgg-11, vgg-13, vgg-16, vgg-19.
    """
    # --- Select backbone ---
    if model_depth == 11:
        backbone = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_depth == 13:
        backbone = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_depth == 19:
        backbone = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
    else:  # default vgg-16
        backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)

    # --- Adapt first conv layer to grayscale ---
    old_conv = backbone.features[0]
    w = old_conv.weight.data.mean(dim=1, keepdim=True)  # average RGB -> 1 channel

    backbone.features[0] = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding
    )
    backbone.features[0].weight.data = w

    # --- Freeze convolutional layers ---
    for p in backbone.features.parameters():
        p.requires_grad = True

    # --- Replace fully-connected classifier with regression head ---
    # NOTE: All VGG variants output a 512×7×7 feature map → 25088 flattened
    backbone.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(True),
        nn.Dropout(0.4),
        nn.Linear(1024, 256),
        nn.ReLU(True),
        nn.Dropout(0.3),
        nn.Linear(256, out_dim)
    )

    return backbone
