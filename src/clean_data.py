# src/clean_data.py

import pandas as pd
import numpy as np
import h5py
from torchvision import transforms
from XRayDataset import XRayDataset
import os

def grab_keys(path, lst):
    with h5py.File(path, 'r') as f:
        lst.extend(list(f.keys()))

def find_data_from_dataframe(img_keys, train_df, val_df, test_df):
    train_df = train_df.loc[train_df.index.intersection(img_keys)]
    val_df = val_df.loc[val_df.index.intersection(img_keys)]
    test_df = test_df.loc[test_df.index.intersection(img_keys)]
    return train_df, val_df, test_df

def shrink_image(arr):
    """
    Shrinks 1024×1024 → 256×256 by averaging 4×4 blocks.
    If you want 512 or 128, resizing is done later in torchvision.
    """
    return arr.reshape(256, 4, 256, 4).mean(axis=(1, 3))

def add_image_data_df(df, file_paths):
    id_set = set(df.index)
    all_images = {}

    for path in file_paths:
        try:
            with h5py.File(path, "r") as file:
                for key in set(file.keys()) & id_set:
                    arr = file[key][()]
                    if isinstance(arr, np.ndarray) and arr.size > 0:
                        arr = shrink_image(arr)   # ↓ still 256×256 here
                        all_images[key] = arr
        except OSError as e:
            print(f"Failed to read {path}: {e}")

    df["img_arr"] = df.index.map(all_images).fillna(np.nan)
    return df

def clean_main(img_size=224):
    """
    Cleans the data and returns train, val, test datasets + mean/std for labels.
    Supports dynamic image resolutions.
    """
    # === 1. File paths ===
    home_dir = os.path.expanduser("~")
    base_path = os.path.join(home_dir, "teams", "b1")

    hdf5_paths = [os.path.join(base_path, "bnpp_frontalonly_1024_0_1.hdf5")] + [
         os.path.join(base_path, f"bnpp_frontalonly_1024_{i}.hdf5") for i in range(1, 11)
    ]

    train_data = os.path.join(base_path, 'BNPP_DT_train_with_ages.csv')
    val_data   = os.path.join(base_path, 'BNPP_DT_val_with_ages.csv')
    test_data  = os.path.join(base_path, 'BNPP_DT_test_with_ages.csv')

    # === 2. collect all HDF5 keys ===
    img_keys = []
    for path in hdf5_paths:
        grab_keys(path, img_keys)

    # === 3. load csv ===
    train_df = pd.read_csv(train_data)
    val_df   = pd.read_csv(val_data)
    test_df  = pd.read_csv(test_data)

    # enforce same columns + unique_key index
    desired_cols = train_df.columns
    train_df = train_df[desired_cols].set_index('unique_key')
    val_df   = val_df[desired_cols].set_index('unique_key')
    test_df  = test_df[desired_cols].set_index('unique_key')

    # === 4. ensure the keys exist in HDF5 ===
    train_df, val_df, test_df = find_data_from_dataframe(
        img_keys, train_df, val_df, test_df
    )

    # === 5. add image arrays ===
    train_df = add_image_data_df(train_df, hdf5_paths)
    val_df   = add_image_data_df(val_df, hdf5_paths)
    test_df  = add_image_data_df(test_df, hdf5_paths)

    # === 6. convert to numpy ===
    X_train = np.stack(train_df['img_arr'])
    X_val   = np.stack(val_df['img_arr'])
    X_test  = np.stack(test_df['img_arr'])

    y_train = train_df['bnpp_value_log'].to_numpy()
    y_val   = val_df['bnpp_value_log'].to_numpy()
    y_test  = test_df['bnpp_value_log'].to_numpy()

    # === 7. standardize labels ===
    y_mean = y_train.mean()
    y_std  = y_train.std()

    y_train = (y_train - y_mean) / y_std
    y_val   = (y_val - y_mean) / y_std
    y_test  = (y_test - y_mean) / y_std

    # === 8. torchvision transform (dynamic!) ===
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

    # === 9. Construct datasets ===
    train_dataset = XRayDataset(X_train, y_train, transform)
    val_dataset   = XRayDataset(X_val, y_val, transform)
    test_dataset  = XRayDataset(X_test, y_test, transform)

    return train_dataset, val_dataset, test_dataset, y_mean, y_std
