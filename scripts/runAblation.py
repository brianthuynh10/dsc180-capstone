import argparse
import pandas as pd
import torch
import os

from models.CNNs.assembleCNN import build_grayscale_cnn
from interpretability.imageOcclusion.ablationPredict import run_ablation_patch_across_image
from config import ABLATION_OUTPUTS_PATH


def main(model_name, patch_size):
    predictions_df = pd.read_csv(
        os.path.join(ABLATION_OUTPUTS_PATH, f"{model_name}_outputs_no_ablation.csv")
    )
    model = build_grayscale_cnn(model_name=model_name)
    predictions_df = predictions_df.set_index("unique_key")
    print("Running runAblation.py!")
    run_ablation_patch_across_image(model, predictions_df, model_name, patch_size)
    print("Script is done running!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation analysis")
    parser.add_argument("model", choices=["resnet50", "vgg16"], help="Model to use")
    parser.add_argument("patch_size", type=int, choices=[32, 64], help="Patch size for ablation")
    args = parser.parse_args()
    main(args.model, args.patch_size)
