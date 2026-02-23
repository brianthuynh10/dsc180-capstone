from models.mediphi.model import MediPhiModel
from models.mediphi.data import extract_response_dict, parse_report
import pandas as pd
import os

def run_mediphi_model():
    """
    Runs the Mediphi model on the given input text and extracts the response dictionary.
    
    Args:
        input_text (str): The input text for the Mediphi model.
    """
    model = MediPhiModel()
    # read in the prompt 
    with open("models/mediphi/edema_prompt.txt", "r") as f:
        prompt = f.read()
    
    # Run the model on sample_data.csv 
    df = pd.read_csv("models/mediphi/sample_data.csv")
    
    # Apply cleaning to 'ReportClean' column
    df['ReportClean'] = df['ReportClean'].apply(parse_report)

    # Generate responses
    df['LLM_output'] = model.make_predictions(prompt, df, BATCH_SIZE=5)

    # Extract response dictionaries
    df['LLM_output'] = df['LLM_output'].apply(extract_response_dict)

    # Save the results to a new CSV file
    os.makedirs("llm_outputs", exist_ok=True)
    df.to_csv("llm_outputs/mediphi_output.csv", index=False)

if __name__ == "__main__":
    run_mediphi_model()

