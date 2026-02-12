# Standard library
import os
import numpy as np
import pandas as pd
import torch
import h5py
from torchvision import transforms


def preprocess_image(img_1024, target_size=256): 
    """
    Resize the input image to the target size while maintaining aspect ratio.
    Use the same target_size the model was trained on.
    """

    def shrink_image(arr, out_size=target_size):
        H, W = arr.shape
        assert H == W
        factor = H // out_size

        img = arr.reshape(out_size, factor, out_size, factor).mean(axis=(1, 3))

        # Min–max normalize per image
        img = img.astype(np.float32)
        min_val = img.min()
        max_val = img.max()

        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)

        return img
    
    img = shrink_image(img_1024)

    # (1, 1, H, W)
    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

    normalize = transforms.Normalize(mean=[0.485], std=[0.229])
    img = normalize(img)

    return img


def create_groups(df: pd.DataFrame):        
    """
    Build dictionary mapping edema presence/severity groups.
    """

    absents = df[df['LLM_Output_Presence'].str.contains('absent', na=False)]
    present = df[df['LLM_Output_Presence'].str.contains('present', na=False)]

    groups = {"absent": absents}

    unique_severities = (
        present['LLM_Output_Severity']
        .dropna()
        .str.lower()
        .str.strip()
        .unique()
    )

    for sev in unique_severities:
        groups[f"present_{sev}"] = present[
            present['LLM_Output_Severity']
            .str.lower()
            .str.contains(str(sev), na=False)
        ]

    return groups


def grab_labeled_bnpp_reports(path='~/teams/b1/bnpp-reports-clean.csv'):
    """
    Load cleaned BNP reports CSV.
    """
    path = os.path.expanduser(path)
    return pd.read_csv(path, usecols=['Phonetic','ReportClean', 'LLM_Output'])


def connect_data(phonetic_name, base="~/teams/b1/"):
    """
    Given a phonetic name, return (hdf5_file_name, key)
    """

    hdf5_names = [f'bnpp_frontalonly_1024_{i}' for i in range(1, 11)]
    hdf5_names.append('bnpp_frontalonly_1024_0_1')

    base = os.path.expanduser(base)

    for name in hdf5_names:
        path = os.path.join(base, f"{name}.hdf5")

        if not os.path.exists(path):
            continue

        with h5py.File(path, "r") as f:
            for key in f.keys():
                if phonetic_name.lower() in key.lower():
                    return (name, key)

    return None
