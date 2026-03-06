import os
import pandas as pd

from interpretability.GradCAM.data import load_connected_reports
from interpretability.GradCAM.GradCAM import GradCAMWrapper
from models.CNNs.assembleCNN import build_grayscale_cnn
from config import BASE_PATH
from tqdm.auto import tqdm


def compare_layers_progression():
    tqdm.write(70 * "=")
    tqdm.write("Running Layer Progression Pipeline (Individual Images)")
    # Phase 1: Data Loading (gather the test set).
    tqdm.write("Starting Phase 1: Loading data...")
    # gather keys in the test set
    test_df = pd.read_csv(
        os.path.join(BASE_PATH, "BNPP_DT_test_with_ages.csv"), usecols=["unique_key"]
    )
    full_reports = load_connected_reports()
    # Make the unique key the name.
    combined = pd.merge(test_df, full_reports, on="unique_key", how="inner")
    tqdm.write("Phase 1 Done")
    # Phase 2: Building Maps
    model_names = ["vgg16", "resnet50"]
    for MODEL_NAME in tqdm(model_names, desc="Models", unit="model"):
        tqdm.write(f"Processing Model: {MODEL_NAME}")
        model = build_grayscale_cnn(model_name=MODEL_NAME)
        # Define the layers to inspect
        if MODEL_NAME == "vgg16":
            layer_dict = {
                "early": model.features[16],
                "mid": model.features[23],
                "final": model.features[-1],
            }
        else:
            layer_dict = {
                "early": model.layer1[-1],
                "mid": model.layer2[-1],
                "final": model.layer4[-1],
            }
        for layer_name, investigate_layer in tqdm(
            list(layer_dict.items()), desc=f"Layers ({MODEL_NAME})", unit="layer", leave=False
        ):
            cam_wrapper = GradCAMWrapper(model=model, target_layer=investigate_layer)
            cam_wrapper.build_averaged_heatmaps(combined, layer_name, model_name=MODEL_NAME)
            tqdm.write(f"✔ CAMS created for {layer_name} on {MODEL_NAME}!")
        tqdm.write(f"✔ ALL CAMS created for {MODEL_NAME}!")
    tqdm.write("Full Pipeline for comparing groups is complete!")


if __name__ == "__main__":
    compare_layers_progression()
