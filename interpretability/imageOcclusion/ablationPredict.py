import torch
from .AblatedBNPPDataset import AblatedBNPPDataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm.auto import tqdm

STD = 0.8821057040979827
AVG = 2.9434068584972755


def run_ablation_patch_across_image(model, predictions_df, model_name, patch_size):
    # create the output path needed
    base = os.path.expanduser(
        f"~/teams/b1/ablation_predictions/{model_name}_ablations_patch_{patch_size}"
    )
    os.makedirs(base, exist_ok=True)

    # setup the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # helper for generating the patches
    def generate_patch_coords(img_size=256, patch_size=patch_size, stride=patch_size):
        coords = []
        for r in range(0, img_size, stride):
            for c in range(0, img_size, stride):
                r1 = r
                r2 = r + patch_size
                c1 = c
                c2 = c + patch_size
                coords.append((r1, r2, c1, c2))
        return coords

    coordinates = generate_patch_coords()
    # loop thru all the possibel cords and create the ablation using mean ablation
    for cord in tqdm(coordinates, desc="Ablating regions", unit="region"):
        dataset = AblatedBNPPDataset(predictions_df, "mean", cord)
        loader = DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )
        # preidctions
        preds = {}

        with torch.no_grad():
            for imgs, image_ids in tqdm(
                loader, desc=f"Predicting {cord}", unit="batch", leave=False
            ):
                imgs = imgs.to(device)

                outputs = model(imgs).view(-1)

                for img_id, pred in zip(image_ids, outputs):
                    preds[img_id] = pred.item() * STD + AVG

        # --------------------------------------------------
        # Convert to DataFrame
        # --------------------------------------------------
        pred_df = pd.DataFrame.from_dict(
            preds, orient="index", columns=["ablation_pred"]
        )
        out_csv_name = f"{model_name}_ablation_predictions_region_{cord}.csv"
        output_path = os.path.join(base, out_csv_name)
        pred_df.to_csv(output_path)

        tqdm.write(f"Ablated predictions saved at {output_path}")
