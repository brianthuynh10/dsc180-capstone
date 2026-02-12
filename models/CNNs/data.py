import os
import h5py
import numpy as np
import pandas as pd
from torchvision import transforms
from XRayDataset import XRayDataset

"""
Data Cleaning:
    - Links HDF5 image data to train / val / test CSVs
    - Downscales images from 1024x1024 → 256x256 via block averaging
    - Standardizes targets using TRAIN statistics only
"""

# --------------------------------------------------
# HDF5 utilities
# --------------------------------------------------

def grab_keys(path, lst):
    with h5py.File(path, "r") as f:
        lst.extend(list(f.keys()))


def find_data_from_dataframe(img_keys, train_df, val_df, test_df):
    """
    Keep only rows whose keys appear in the HDF5 files
    """
    train_df = train_df.loc[train_df.index.intersection(img_keys)]
    val_df = val_df.loc[val_df.index.intersection(img_keys)]
    test_df = test_df.loc[test_df.index.intersection(img_keys)]
    return train_df, val_df, test_df


# --------------------------------------------------
# Image processing
# --------------------------------------------------

def shrink_image(arr, out_size=256):
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

def add_image_data_df(df, file_paths, image_size=256):
    id_set = set(df.index)
    all_images = {}

    for path in file_paths:
        try:
            with h5py.File(path, "r") as file:
                for key in set(file.keys()) & id_set:
                    arr = file[key][()]
                    if isinstance(arr, np.ndarray) and arr.size > 0:
                        all_images[key] = shrink_image(arr, image_size)
        except OSError as e:
            print(f"Failed to read {path}: {e}")

    df["img_arr"] = df.index.map(all_images)
    return df

# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main(image_size=256):
    """Clean data and return datasets + normalization stats"""
    print(f"Using image size: {image_size}×{image_size}")
    # Paths
    home_dir = os.path.expanduser("~")
    base_path = os.path.join(home_dir, "teams", "b1")

    hdf5_paths = [os.path.join(base_path, "bnpp_frontalonly_1024_0_1.hdf5")] + [
        os.path.join(base_path, f"bnpp_frontalonly_1024_{i}.hdf5") for i in range(1, 11)
    ]

    train_csv = os.path.join(base_path, "BNPP_DT_train_with_ages.csv")
    val_csv   = os.path.join(base_path, "BNPP_DT_val_with_ages.csv")
    test_csv  = os.path.join(base_path, "BNPP_DT_test_with_ages.csv")


    # Step 1: collect all image keys
    print("Clean Data Step 1: Collecting HDF5 keys")
    img_keys = []
    for path in hdf5_paths:
        grab_keys(path, img_keys)

    # Step 2: load CSVs
    print("Clean Data Step 2: Loading CSVs")
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    desired_cols = train_df.columns
    train_df = train_df[desired_cols].set_index("unique_key")
    val_df   = val_df[desired_cols].set_index("unique_key")
    test_df  = test_df[desired_cols].set_index("unique_key")
    
    # Step 3: align CSVs with available images
    print("Clean Data Step 3: Aligning data")
    train_df, val_df, test_df = find_data_from_dataframe(
        img_keys, train_df, val_df, test_df
    )

    # Step 4: attach image arrays
    print("Clean Data Step 4: Loading and downsampling images")
    train_df = add_image_data_df(train_df, hdf5_paths,image_size)
    val_df   = add_image_data_df(val_df, hdf5_paths, image_size)
    test_df  = add_image_data_df(test_df, hdf5_paths, image_size)


    # Step 5: stack arrays
    print("Clean Data Step 5: Stacking arrays")
    X_train = np.stack(train_df["img_arr"].values)
    X_val   = np.stack(val_df["img_arr"].values)
    X_test  = np.stack(test_df["img_arr"].values)

    # Step 6: standardize targets (TRAIN stats only)
    y_train = train_df["bnpp_value_log"].to_numpy()
    y_val   = val_df["bnpp_value_log"].to_numpy()
    y_test  = test_df["bnpp_value_log"].to_numpy()

    y_mean = y_train.mean()
    y_std  = y_train.std()

    y_train = (y_train - y_mean) / y_std
    y_val   = (y_val   - y_mean) / y_std
    y_test  = (y_test  - y_mean) / y_std

    # Step 7: minimal transforms (no resizing!)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    print("Creating XRayDataset objects...")
    train_dataset = XRayDataset(X_train, y_train, transform)
    val_dataset   = XRayDataset(X_val,   y_val,   transform)
    test_dataset  = XRayDataset(X_test,  y_test,  transform)
    print("Data Pre-Processing Complete!")

    return train_dataset,  val_dataset, test_dataset, y_mean, y_std

if __name__ == "__main__":
    main()

