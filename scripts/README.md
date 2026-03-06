# Scripts

This folder contains all runnable pipelines for our DSC180 Capstone experiments, including:

- CNN training (VGG16 and ResNet50)
- Image ablation experiments
- Grad-CAM interpretability analysis
- Inference using our fine-tuned MediPhi model

Each script is designed to be executed as a Python module from the root of the repository.

---

## Getting Started

Before running any scripts:

1. Navigate to the root of the project directory.  
   Your terminal should look something like:

   ```
   private/dsc180-capstone/
   ```

2. Run scripts as modules using the `-m` flag.  
   Do **not** include the `.py` extension.

### General Command Format

```bash
python3 -m scripts.<script_name>
```

### Example

```bash
python3 -m scripts.runAblation
```

---

## 📂 Script Descriptions

---

### `mediphi.py`
**Command:**
```bash
python3 -m scripts.mediphi
```

<b> Prerequisites: </b>
If you are running inside DSMLP (without a custom Docker pod), the default environment contains package versions that are incompatible with the MediPhi model and LoRA adapter. To combat that, run these commands below to use the MediPhi model with LoRA adapters.

```
pip uninstall -y torch torchvision torchaudio transformers peft accelerate numpy
# Next, 
pip install \
  numpy==1.26.4 \
  torch==2.3.1 \
  torchvision==0.18.1 \
  torchaudio==2.3.1 \
  transformers==4.44.2 \
  peft \
  accelerate==0.30.1
```

Assembles and runs inference using our fine-tuned Microsoft MediPhi model trained on radiology reports.

**What it does:**
- Loads the fine-tuned MediPhi checkpoint
- Runs predictions on a small batch of example reports included in the repository
- Outputs model predictions for inspection

**Use case:**  
Quick validation of the fine-tuned language model on report-level data.

---

### `runAblation.py`

Runs our image ablation experiment on the test set.

**Command:**
```bash
python3 -m scripts.runAblation <model> <patch_size>
```
- `<model>`: Choose `resnet50` or `vgg16`
- `<patch_size>`: Choose `32` or `64`

**Example:**
```bash
python3 -m scripts.runAblation resnet50 32
```

**What it does:**
- Applies localized patch ablations to X-ray images
- Runs ablated images through the specified CNN model (our experiments are on image patch size 32)
- Records predictions for each ablated image
- Saves outputs as a CSV file (including image ID and predicted value)

**Use case:**  
Quantifying model sensitivity to specific spatial regions of the image.

---

### `runGradCAM.py`

**Command:**
```bash
python3 -m scripts.runGradCAM
```

Runs the Grad-CAM interpretability pipeline across grouped patient categories.

**What it does:**
- Groups patients by edema severity
- Runs Grad-CAM on each group
- Averages gradient maps across the group
- Converts averaged gradients into heatmaps
- Overlays heatmaps onto representative X-ray images
- Saves visualizations to the `outputs/` directory

**Use case:**  
Understanding where CNN models focus when making regression predictions.

---

### `train_cnn.py`

Trains a CNN model (ResNet50 or VGG16) for X-ray image regression (predicting log-BNPP).

**Command:**
```bash
python3 -m scripts.train_cnn <model>
```
- `<model>`: Choose `resnet50` or `vgg16`

**Examples:**
```bash
python3 -m scripts.train_cnn resnet50
python3 -m scripts.train_cnn vgg16
```

**What it does:**
- Loads training and validation data
- Trains the specified CNN model for regression
- Logs training metrics to Weights & Biases
- Saves:
  - Latest checkpoint (after each epoch)
  - Best-performing model (based on validation metric)
- Outputs saved under the designated `outputs/` folder

**Important:**  
Training may take 1–2 hours depending on GPU availability.  
We recommend running this in a background pod (e.g., DSMLP GPU instance).

