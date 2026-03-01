from interpretability.GradCAM.data import create_groups
from interpretability.GradCAM.GradCAM import GradCAMWrapper
from models.CNNs.assembleCNN import build_grayscale_cnn
from interpretability.GradCAM.BNPPDataset import BNPPDataset
import os 
from torch.utils.data import DataLoader
import pandas as pd
import torch
import matplotlib.pyplot as plt
import ast
import numpy as np


def batch_predict(model, df, batch_size=16, device="cuda"):
    df = df.reset_index(drop=True)
    STD = 0.8821057040979827
    AVG = 2.9434068584972755
    base = os.path.expanduser('~/teams/b1/')
    model.eval().to(device)
    dataset = BNPPDataset(df, base)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,      
        pin_memory=True     
    )
    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(device)        # (B, 1, H, W)
            out = model(imgs).squeeze(0)   # (B,)
            out = out * STD + AVG
            for i, pred in zip(idxs, out.cpu().numpy()):
                df.at[i.item(), "bnpp_prediction"] = pred.item()
    return df

def compare_layers():
    print(70 * '=')
    print("Running the Pipeline to Compare Groups")
    # Data
    print('Starting Phase 1: Loading in data...')
    full_reports = pd.read_csv("~/teams/b1/processed_data/connected_bnpp_reports.csv")
    full_reports.dropna(subset=['hdf5_file_name'], inplace=True)
    full_reports = full_reports.sample(20, random_state=42)
    full_reports['hdf5_file_name'] = full_reports['hdf5_file_name'].apply(ast.literal_eval)
    print('Phase 1 Done')
    # looping through each model to find activations within the groups
    print('Starting Phase 2: Buiding Maps for Models')
    model_names = ['vgg16', 'resnet50']
    for MODEL_NAME in model_names:
        print(f"\n Loading Model: {MODEL_NAME}")
        # Loop creates maps across all groups for ONE model 
        all_results = []
        model = build_grayscale_cnn(model_name=MODEL_NAME)
        if MODEL_NAME == 'vgg16':
            layer_dict = {
                "early": model.features[16],
                "mid": model.features[23],
                "final": model.features[-1]
            }
        else:
            layer_dict = {
                "early": model.layer1[-1],
                "mid": model.layer2[-1],
                "final": model.layer4[-1]
            }
        # Predictions: 
        print('Generating Predictions...')
        df_preds = batch_predict(model, full_reports, batch_size=32)
        for layer_name, model_layer in layer_dict.items():
            print(f'Finding Avg Gradients for {layer_name} layer')
            cam = GradCAMWrapper(model, model_layer)
            avg_cam = cam.generate_average_cam(df_preds)
            print(f"✔ Avg CAM shape created!")
            all_results.append({ 
                "model": MODEL_NAME,
                "layer": layer_name,
                "avg_cam" : avg_cam
            })
            print(f'Built CAM for {layer_name} layer')
        print('Completed building CAM maps')

        heat_builder = GradCAMWrapper(model, target_layer=layer_dict['final']) # the layer won't matter because we're only build a map
        stacked = np.stack([r['avg_cam'] for r in all_results])
        model_min = np.percentile(stacked, 50)
        model_max = np.percentile(stacked, 99)

        out_path = out_path = os.path.join('Layer_GradCAM_Outputs', f'{MODEL_NAME}_maps')
        os.makedirs(out_path, exist_ok=True)
        for layer in all_results: 
            model_name = layer['model']
            layer_name = layer['layer']
            avg_cam = layer['avg_cam']

            heatmap = heat_builder.build_averaged_cam_map(average_cam=avg_cam,
                                                 grouped_df=full_reports,
                                                 global_min=model_min,
                                                 global_max=model_max)
            save_path = os.path.join(out_path, f'{model_name}_{layer_name}_cam.png')
            plt.imsave(save_path, heatmap)
            print(f'Saved GradCAM heatmap at {save_path}')
        print(f"Processed finish for {MODEL_NAME}")
    print("Full Pipeline for comparing layers is complete!") 

def compare_groups(): 
     # Data
    print('Starting Phase 1: Loading in data...')
    full_reports = pd.read_csv("~/teams/b1/processed_data/connected_bnpp_reports.csv")
    full_reports.dropna(subset=['hdf5_file_name'], inplace=True)
    full_reports['hdf5_file_name'] = full_reports['hdf5_file_name'].apply(ast.literal_eval)
    grouped_dict = create_groups(full_reports)
    print('Phase 1 Done')
    print(70 * '=')
    # looping through each model to find activations within the groups
    print('Starting Phase 2: Buiding Maps for Models')
    model_names = ['vgg16', 'resnet50']
    for MODEL_NAME in model_names:
        print(f"\n Loading Model: {MODEL_NAME}")
        # Loop creates maps across all groups for ONE model 
        all_results = []
        model = build_grayscale_cnn(model_name=MODEL_NAME)
        # last layers of the models
        if MODEL_NAME == 'vgg16':
            TARGET_LAYER = model.features[-1]
        elif MODEL_NAME == 'resnet50':
            TARGET_LAYER = model.layer4[-1]
        
        cam = GradCAMWrapper(model=model, target_layer=TARGET_LAYER)

        for group_name, group_df in grouped_dict.items():
            print(f"Working on: {group_name}")
            group_df = group_df.sample(10, random_state=42)
            print("Generating predictions...")
            group_df_pred = batch_predict(model, group_df, batch_size=32)
            print("Computing 10 CAM Heatmaps...")
            cam.generate_10_cam(group_df_pred, group_name, MODEL_NAME)
            print(f"✔ 10 CAMS created for {group_name}!")
            
        print("Finished generating all RAW CAMs")
        print("=" * 70)
        print(f"Processed finish for {MODEL_NAME}")
    print("Full Pipeline for comparing groups is complete!") 

if __name__ == "__main__": 
    compare_groups()