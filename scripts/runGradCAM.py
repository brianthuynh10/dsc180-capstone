from interpretability.GradCAM.data import grab_labeled_bnpp_reports, connect_data, preprocess_image, create_groups
from interpretability.GradCAM.GradCAM import GradCAM, RegressionTarget
from models.vgg.buildModel import build_grayscale_vgg16, load_model, generate_prediction
import pandas as pd
import torch
import matplotlib.pyplot as plt
import h5py
import os

def batch_predict(model, df, batch_size=16, device="cuda"):
    train_df = pd.read_csv('/home/bth001/teams/b1/BNPP_DT_val_with_ages.csv')
    avg = train_df['bnpp_value_log'].mean()
    std = train_df['bnpp_value_log'].std()

    base = os.path.expanduser('~/teams/b1/')
    model.eval().to(device)

    preds = []

    batch_imgs = []
    batch_indices = []

    for idx, row in df.iterrows():
        hdf5_file, hdf5_key = row['hdf5_file_name']
        hdf5_path = os.path.join(base, hdf5_file + ".hdf5")

        with h5py.File(hdf5_path, "r") as f:
            img = f[hdf5_key][()]

        img_tensor = preprocess_image(img)  # [1, C, H, W]
        batch_imgs.append(img_tensor)
        batch_indices.append(idx)

        if len(batch_imgs) == batch_size:
            batch = torch.cat(batch_imgs).to(device)

            with torch.no_grad():
                out = model(batch).squeeze()
                out = out * std + avg

            for i, pred in zip(batch_indices, out.cpu().numpy()):
                df.at[i, "bnpp_prediction"] = float(pred)

            batch_imgs.clear()
            batch_indices.clear()

    # leftover
    if batch_imgs:
        batch = torch.cat(batch_imgs).to(device)
        with torch.no_grad():
            out = model(batch).squeeze()
            out = out * std + avg

        for i, pred in zip(batch_indices, out.cpu().numpy()):
            df.at[i, "bnpp_prediction"] = float(pred)

    return df



def main(): 
    ### Phase 1: 1) Load Data , 2) Load Model , 3) Build Groups ### 
    full_reports = pd.read_csv("~/teams/b1/processed_data/connected_bnpp_reports.csv")
    model = build_grayscale_vgg16()
    model = load_model(model)
    grouped_dict = create_groups(full_reports)

    ### Phase 2: Generate Predictions to help with GradCAM ###
    for group_name, group_df in grouped_dict.items(): # eary stop for now
        group_df = batch_predict(model, group_df)
        grouped_dict[group_name] = group_df

    ### Phase 3: Generate CAM maps for each image and build average CAM maps for each group ###
    for group_name, group_df in grouped_dict.items():
        gradCAM = GradCAM(model=model, target_layer=model.features[-1])
        average_cam_map = gradCAM.generate_average_cam(group_df)
        heatmap = gradCAM.build_averaged_cam_map(average_cam_map, group_df)
        # save the image to disk 
        plt.imsave(f"outputs/{group_name}_average_cam.png", heatmap)
if __name__ == "__main__":
    main()