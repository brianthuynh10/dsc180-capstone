import os
import torch
import numpy as np
import h5py
import pandas as pd
import cv2
from pytorch_grad_cam import GradCAM
from .data import preprocess_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import BASE_PATH


class RegressionTarget:
    def __call__(self, model_output):
        return model_output.squeeze()


class GradCAMWrapper:
    def __init__(self, model, target_layer, alpha_min=0.02, alpha_max=0.45):
        self.model = model
        self.target_layer = target_layer
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        # create GradCAM ONCE
        self.cam = GradCAM(model=self.model, target_layers=[self.target_layer])

    def _generate_cam(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        grayscale_cam = self.cam(
            input_tensor=input_tensor, targets=[RegressionTarget()]
        )
        cam_map = grayscale_cam[0]  # (H, W)
        cam_map = np.maximum(cam_map, 0)
        return cam_map

    def _overlay_cam(self, cam, img):
        alpha_map = self.alpha_min + (self.alpha_max - self.alpha_min) * cam[..., None]
        heatmap_bgr = (
            cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET).astype(np.float32)
            / 255.0
        )
        heatmap_rgb = (
            cv2.cvtColor(
                (heatmap_bgr * 255).astype(np.uint8), cv2.COLOR_BGR2RGB
            ).astype(np.float32)
            / 255.0
        )
        img_rgb = np.stack([img, img, img], axis=-1)
        overlay = alpha_map * heatmap_rgb + (1 - alpha_map) * img_rgb
        return np.clip(overlay, 0, 1)

    def build_heatmaps(
        self,
        job_name: str,
        grouped_df: pd.DataFrame,
        cur_group: str,
        model_name: str,
        target_size=256,
        global_min=None,
        global_max=None,
    ):
        out_path = os.path.join(
            f"{job_name}_output", f"{model_name}_maps", f"{cur_group}"
        )
        os.makedirs(out_path, exist_ok=True)

        # Select 10 random samples (reproducible)
        sampled_df = grouped_df.sample(n=10, random_state=0)

        for count, (_, row) in enumerate(sampled_df.iterrows()):

            hdf5_file, hdf5_key = row["hdf5_file_name"]
            hdf5_path = os.path.join(BASE_PATH, hdf5_file + ".hdf5")

            with h5py.File(hdf5_path, "r") as f:
                img_1024 = f[hdf5_key][()]

            img_tensor = preprocess_image(img_1024, target_size)

            # Generate raw CAM
            cam_map = self._generate_cam(
                img_tensor
            )  # gets the model prediciton and genereate cam map

            # Apply global normalization if provided
            if global_min is not None and global_max is not None:
                cam_map = np.clip(cam_map, global_min, global_max)
                cam_map = (cam_map - global_min) / (global_max - global_min + 1e-8)

                # Optional blob shaping
                cam_map = cam_map**0.3
            # Prepare image for overlay
            img = img_tensor[0, 0].detach().cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            cam_map = cv2.resize(
                cam_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC
            )
            # Build overlay
            overlay = self._overlay_cam(cam_map, img)
            save_path = os.path.join(out_path, f"{cur_group}_{count}.png")
            plt.imsave(save_path, overlay)
            print(f"Saved overlay: {save_path}")

    def build_averaged_heatmaps(
        self,
        test_set_df,
        layer: str,
        model_name: str,
        target_size=256,
        global_min=None,
        global_max=None,
    ):
        out_path = os.path.join("GradCAM_Layer", model_name)
        os.makedirs(out_path, exist_ok=True)
        sum_img = None
        sum_cam = None
        n = 0
        for _, row in tqdm(
            test_set_df.iterrows(),
            total=len(test_set_df),
            desc=f"Processing GradCAM for {model_name}'s {layer} layer",
        ):
            hdf5_file, hdf5_key = row["hdf5_file_name"]
            hdf5_path = os.path.join(BASE_PATH, hdf5_file + ".hdf5")

            with h5py.File(hdf5_path, "r") as f:
                img_1024 = f[hdf5_key][()]

            img_tensor = preprocess_image(img_1024, target_size)

            cam_map = self._generate_cam(img_tensor)

            if global_min is not None and global_max is not None:
                cam_map = np.clip(cam_map, global_min, global_max)
                cam_map = (cam_map - global_min) / (global_max - global_min + 1e-8)
                cam_map = cam_map**0.3

            img = img_tensor[0, 0].detach().cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            cam_map = cv2.resize(
                cam_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC
            )

            if sum_img is None:
                sum_img = np.zeros_like(img)
                sum_cam = np.zeros_like(cam_map)

            sum_img += img
            sum_cam += cam_map
            n += 1
        avg_img = sum_img / n
        avg_cam = sum_cam / n
        avg_cam = avg_cam / (avg_cam.max() + 1e-8)
        overlay = self._overlay_cam(avg_cam, avg_img)
        save_path = os.path.join(out_path, f"{layer}_average.png")
        plt.imsave(save_path, overlay)
        print(f"Saved averaged overlay: {save_path}")
