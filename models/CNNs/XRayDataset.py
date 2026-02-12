import torch
import numpy as np

class XRayDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        """
        images: numpy array (N, 256, 256)
        labels: numpy array (N,)
        transform: optional torch transform (expects tensor)
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # (256,256) → (1,256,256)
        image = torch.from_numpy(self.images[idx]).float().unsqueeze(0)

        # DO NOT normalize per image
        # Any normalization should already be in `transform`
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32).view(1)
        return image, label
