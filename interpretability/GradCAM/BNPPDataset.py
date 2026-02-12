import os
import torch
import h5py
from .data import preprocess_image

class BNPPDataset(torch.utils.data.Dataset):
    def __init__(self, df, base):
        self.df = df.reset_index(drop=True)
        self.base = base

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        hdf5_file, hdf5_key = self.df.loc[idx, "hdf5_file_name"]
        path = os.path.join(self.base, hdf5_file + ".hdf5")

        with h5py.File(path, "r") as f:
            img = f[hdf5_key][()]

        img_tensor = preprocess_image(img)
        
        if img_tensor.ndim == 4:
            img_tensor = img_tensor.squeeze(0)

        assert img_tensor.shape == (1, 256, 256), img_tensor.shape

        return img_tensor, idx
