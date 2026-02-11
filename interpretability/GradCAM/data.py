import pandas as pd 
import numpy as np 
import regex as re
import h5py as h5
import os
import torch
from torchvision import transforms

def preprocess_image(img_1024, target_size=256): 
    """
    Resize the input image to the target size while maintaining aspect ratio.
    - This comes from the data.py file in the VGG pipeline, use this for generating the images for GradCAM
    - Be sure to use the same tartget size as the model was trained on (i.e. if you trained on 256x256, use target_size=256 here)
    Args:
        image (PIL.Image): Input image to be resized.
        target_size (int): Desired size for the shorter side of the image.
    """
    def shrink_image(arr, out_size=target_size):
        H, W = arr.shape
        assert H == W
        factor = H // out_size

        img = arr.reshape(out_size, factor, out_size, factor).mean(axis=(1, 3))

        # --- min–max normalize per image ---
        img = img.astype(np.float32)
        min_val = img.min()
        max_val = img.max()

        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)

        return img
    
    # 1. Downsample + per-image min–max normalize
    img = shrink_image(img_1024)        # (256, 256), float32 in [0,1]

    # 2. Numpy → torch tensor, add channel dim
    img = torch.from_numpy(img).float().unsqueeze(0)  # (1, 256, 256)

    # 3. Apply SAME normalization as training
    normalize = transforms.Normalize(mean=[0.485], std=[0.229])
    img = normalize(img)
    return img

def create_groups(df: pd.DataFrame):        
    """
    Builds a dictionary mapping where keys are the type of unique edema labels and absenet groups. Ex: 
        - Absent group 
        - present_mild group
        - present_moderate group 
    Args:
    df (pd.DataFrame): The input DataFrame containing the 'LLM_Output' column with edema labels.
    """
    absents = df[df['LLM_Output_Presence'].str.contains('absent', na=False)]
    present = df[df['LLM_Output_Presence'].str.contains('present', na=False)]

    groups = {"absent": absents}

    unique_severities = present['LLM_Output_Severity'].dropna().unique()
    for sev in unique_severities:
        groups[f"present_{sev}"] = present[
            present['LLM_Output_Severity'].str.contains(str(sev), na=False)
        ]
    return groups


def grab_labeled_bnpp_reports(path='~/teams/b1/bnpp-reports-clean.csv'):
    """
    Load cleaned BNP reports CSV and return DataFrame
    """
    return pd.read_csv(path, usecols=['Phonetic','ReportClean', 'LLM_Output'])

def connect_data(phonetic_name, base="~/teams/b1/"):
    """
    Given a phonetic name, return the HDF5 file name that contains it
    and associated key for faster lookup later
    """
    hdf5_names = [f'bnpp_frontalonly_1024_{i}' for i in range(1,11)]
    hdf5_names.append('bnpp_frontalonly_1024_0_1')

    base = os.path.expanduser(base)

    for name in hdf5_names:
        path = os.path.join(base, f"{name}.hdf5")

        if not os.path.exists(path):
            continue

        with h5py.File(path, "r") as f:
            for key in f.keys():
                if phonetic_name.lower() in key.lower():
                    return (name, key) # return the hdf5 file name and the key within that file that contains the image data for the given phonetic name

    return None

