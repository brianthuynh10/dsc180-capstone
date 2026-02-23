import torch
import pandas as pd
from models.CNNs.assembleCNN import build_grayscale_cnn
from interpretability.imageOcclusion.ablationPredict import run_ablation_patch_across_image
    
def main():
    # EDIT THESE HERE
    MODEL_NAME = "vgg16" # 'resnet50' or 'vgg16' 
    PATCH_SIZE = 64 # 32 or 64 (this is what we used)

    # -- DO NOT TOUCH BELOW -- # 
    predictions_df = pd.read_csv(
        f"~/teams/b1/ablation_predictions/{MODEL_NAME}_outputs_no_ablation.csv" # <- edit path if needed
    )
    model = build_grayscale_cnn(model_name=MODEL_NAME)
    predictions_df = predictions_df.set_index("unique_key")
    print('Running runAblation.py!')
    run_ablation_patch_across_image(model, predictions_df, MODEL_NAME, PATCH_SIZE)
    print('Script is done Running!')


if __name__ == "__main__":
    main()