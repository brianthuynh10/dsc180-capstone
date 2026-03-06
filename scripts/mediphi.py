import os

import pandas as pd

from models.mediphi.data import extract_response_dict, parse_report
from models.mediphi.model import MediPhiModel


def run_mediphi_model():
    """Run the MediPhi model and save structured outputs to CSV."""
    model = MediPhiModel()
    # Read in the prompt.
    with open("models/mediphi/edema_prompt.txt", "r") as f:
        prompt = f.read()
    # Run the model on sample_data.csv.
    df = pd.read_csv("models/mediphi/sample_data.csv")
    # Apply cleaning to "ReportClean" column.
    df["ReportClean"] = df["ReportClean"].apply(parse_report)
    # Generate responses.
    df['LLM_output'] = model.make_predictions(prompt, df, 5)
    # Extract response dictionaries.
    df["LLM_output"] = df["LLM_output"].apply(extract_response_dict)
    # Save the results to a new CSV file.
    os.makedirs("llm_outputs", exist_ok=True)
    df.to_csv("MEDIPHI_OUTPUTS/mediphi_output.csv", index=False)


if __name__ == "__main__":
    run_mediphi_model()
