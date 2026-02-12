import os
import torch
import numpy as np
import h5py
import pandas as pd
import cv2
from pytorch_grad_cam import GradCAM
from .data import preprocess_image


class RegressionTarget:
    def __call__(self, model_output):
        return model_output.squeeze()
    
class GradCAMWrapper: 
    def __init__(self, model, target_layer, alpha_min=0.15, alpha_max=0.55):
        self.model = model
        self.target_layer = target_layer
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

        # create GradCAM ONCE
        self.cam = GradCAM(
            model=self.model,
            target_layers=[self.target_layer]
        )

    def _generate_cam(self, input_tensor):
        input_tensor = input_tensor.to(self.device)

        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=[RegressionTarget()]
        )

        cam_map = grayscale_cam[0]  # (H, W)
        cam_map = np.maximum(cam_map, 0)
        cam_map = cam_map ** 0.5    # you can tune this

        return cam_map

    def _overlay_cam(self, cam, img):
        alpha_map = self.alpha_min + (self.alpha_max - self.alpha_min) * cam[..., None]

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_TURBO
        ).astype(np.float32) / 255.0

        img_rgb = np.stack([img, img, img], axis=-1)
        overlay = alpha_map * heatmap + (1 - alpha_map) * img_rgb
        return np.clip(overlay, 0, 1)

    def build_cam_map(self, cam, img_tensor):
        img = img_tensor[0, 0].detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        cam = self._normalize_cam(cam, p_low=30, p_high=99)
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        cam = cv2.GaussianBlur(cam, (5, 5), 0)

        return self._overlay_cam(cam, img)

    def generate_average_cam(self, grouped_df: pd.DataFrame, target_size=256):
        cam_maps = []
        base_path = os.path.expanduser("~/teams/b1/")

        for _, row in grouped_df.iterrows():
            try: 
                hdf5_file, hdf5_key = row['hdf5_file_name']
                hdf5_path = os.path.join(base_path, hdf5_file + ".hdf5")
            except: 
                print(f'{row} was a bad')

            with h5py.File(hdf5_path, "r") as f:
                img_1024 = f[hdf5_key][()]

            img_tensor = preprocess_image(img_1024, target_size)
            cam_map = self._generate_cam(img_tensor)
            cam_maps.append(cam_map)

        cam_maps = np.stack(cam_maps, axis=0)
        return cam_maps.mean(axis=0)

    def build_averaged_cam_map(self, average_cam, grouped_df, target_size=256):
        base_path = os.path.expanduser("~/teams/b1/")
        hdf5_file, hdf5_key = grouped_df.iloc[0]["hdf5_file_name"]

        with h5py.File(os.path.join(base_path, hdf5_file + ".hdf5"), "r") as f:
            img_1024 = f[hdf5_key][()]

        img_tensor = preprocess_image(img_1024, target_size)
        img = img_tensor[0, 0].detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        average_cam = self._normalize_cam(average_cam, p_low=30, p_high=99)
        average_cam = cv2.resize(
            average_cam,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

        return self._overlay_cam(average_cam, img)

    def _normalize_cam(self, cam, p_low=10, p_high=95):
        lo = np.percentile(cam, p_low)
        hi = np.percentile(cam, p_high)
        cam = np.clip(cam, lo, hi)
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam
