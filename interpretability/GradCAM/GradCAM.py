import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import random
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from .data import preprocess_image 
import os

class RegressionTarget:
    def __call__(self, model_output):
        return model_output.squeeze()
    
class GradCAM: 
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.cam = GradCAM(model=self.model, target_layers=[self.target_layer], use_cuda=torch.cuda.is_available())

    def _generate_cam(self, input_tensor):
        """
        Generate a CAM map for a single input image tensor.
        
        :param self: Description
        :param input_tensor: Description
        """
        input_tensor = input_tensor.to(device='cuda' if torch.cuda.is_available() else 'cpu')
        cam = GradCAM(
        model= self.model,
            target_layers=[self.target_layer]
            )
    
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[RegressionTarget()]
        )

        cam_map = grayscale_cam[0]  # (H, W)
        cam_map = np.maximum(cam_map, 0)
        cam_map = cam_map ** 0.5

        return cam_map
    
    def build_cam_map(self, cam, img_tensor, alpha=0.25):
        img = img_tensor[0, 0].detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        cam = self.normalize_cam(cam, p_low=30, p_high=99)  
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        cam = cv2.GaussianBlur(cam, (5, 5), 0)

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_TURBO      # 🔥 CHANGE FROM JET
        ).astype(np.float32) / 255.0

        img_rgb = np.stack([img, img, img], axis=-1)
        overlay = alpha * heatmap + (1 - alpha) * img_rgb
        return np.clip(overlay, 0, 1)


    def generate_average_cam(self, grouped_df: pd.DataFrame, target_size=256):
        """
        Builds an average CAM map for ONE group (i.e. absent, present_mild, present_moderate, etc.) by averaging the CAM maps of all images in that DataFrame
        """
        cam_maps = np.array([])  # list to store CAM maps for each image in the group
        base_path = "~/teams/b1/"
        for idx, row in grouped_df.iterrows(): 
            with h5py.File(base_path + row['hdf5_file'][0], 'r') as f: 
                img_1024 = f[row['hdf5_file'][1]][()]  # load the 1024x1024 image from the hdf5 file using the key
                img_tensor = preprocess_image(img_1024, target_size)  # resize and normalize the image to the target size (e.g. 256x256) and convert to tensor
                cam_map = self._generate_cam(img_tensor)  # generate the CAM map for this image
                cam_maps = np.append(cam_maps, cam_map)
        average_cam = np.mean(cam_maps, axis=0)  # average the CAM maps across all images in the group
        return average_cam
    
    def build_averaged_cam_map(self, average_cam, grouped_df, target_size=256, alpha=0.25):
        base_path = os.path.expanduser("~/teams/b1/")
        hdf5_file, hdf5_key = grouped_df.iloc[0]["hdf5_file_name"]

        with h5py.File(os.path.join(base_path, hdf5_file+'.hdf5'), "r") as f:
            img_1024 = f[hdf5_key][()]

        img_tensor = preprocess_image(img_1024, target_size)

        img = img_tensor[0, 0].detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        average_cam = self.normalize_cam(average_cam, p_low=30, p_high=99)  
        average_cam = cv2.resize(average_cam, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        average_cam = cv2.GaussianBlur(average_cam, (5, 5), 0)

        heatmap = cv2.applyColorMap(
            np.uint8(255 * average_cam),
            cv2.COLORMAP_TURBO      # 🔥 CHANGE
        ).astype(np.float32) / 255.0

        img_rgb = np.stack([img, img, img], axis=-1)
        overlay = alpha * heatmap + (1 - alpha) * img_rgb
        return np.clip(overlay, 0, 1)


    def normalize_cam(cam, p_low=30, p_high=99):
        lo = np.percentile(cam, p_low)
        hi = np.percentile(cam, p_high)
        cam = np.clip(cam, lo, hi)
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam

    
