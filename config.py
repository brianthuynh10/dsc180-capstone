import os

# Base path for all data and models
BASE_PATH = os.path.expanduser("~/teams/b1/")

# Subdirectories
PROCESSED_DATA_PATH = os.path.join(BASE_PATH, "processed_data")
MODEL_WEIGHTS_PATH = os.path.join(BASE_PATH, "cnn_model_weights")
ABLATION_OUTPUTS_PATH = os.path.join(BASE_PATH, "ablation_predictions")