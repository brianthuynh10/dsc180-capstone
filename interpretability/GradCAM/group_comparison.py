from interpretability.GradCAM.data import create_groups, load_connected_reports
from interpretability.GradCAM.GradCAM import GradCAMWrapper
from models.CNNs.assembleCNN import build_grayscale_cnn
from tqdm.auto import tqdm


def compare_groups():
    tqdm.write("Starting Phase 1: Loading in data...")
    full_reports = load_connected_reports()
    grouped_dict = create_groups(full_reports)
    tqdm.write("Phase 1 Done")
    tqdm.write(70 * "=")
    tqdm.write("Starting Phase 2: Building Maps for Models")
    model_names = ["vgg16", "resnet50"]
    for MODEL_NAME in tqdm(model_names, desc="Models", unit="model"):
        tqdm.write(f"Loading Model: {MODEL_NAME}")
        model = build_grayscale_cnn(model_name=MODEL_NAME)
        # last layers of the models
        if MODEL_NAME == "vgg16":
            TARGET_LAYER = model.features[-1]
        elif MODEL_NAME == "resnet50":
            TARGET_LAYER = model.layer4[-1]
        cam = GradCAMWrapper(model=model, target_layer=TARGET_LAYER)
        for group_name, group_df in tqdm(
            list(grouped_dict.items()), desc=f"Groups ({MODEL_NAME})", unit="group", leave=False
        ):
            group_df = group_df.sample(10, random_state=0)
            tqdm.write(f"Working on: {group_name}")
            cam.build_heatmaps("Groups", group_df, group_name, MODEL_NAME)
            tqdm.write(f"✔ 10 CAMS created for {group_name}!")
        tqdm.write("Finished generating all RAW CAMs")
        tqdm.write("=" * 70)
        tqdm.write(f"Processed finish for {MODEL_NAME}")
    tqdm.write("Full Pipeline for comparing groups is complete!")


if __name__ == "__main__":
    compare_groups()
