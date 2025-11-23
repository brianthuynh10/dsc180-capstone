#!/bin/bash
set -e

# List of resolutions
RESOLUTIONS=(128 256 512)

for R in "${RESOLUTIONS[@]}"; do
  LOGFILE="run_res_${R}.log"
  echo "=== STARTING RESOLUTION ${R} ===" | tee -a "$LOGFILE"

  python3 main.py \
    --img_size "$R" \
    --epochs 50 \
    --batch_size 16 \
    --wandb \
    --wandb_project "resnet-resolution-study" \
    --run_name "res_${R}_run" \
    --gpu > "$LOGFILE" 2>&1

  echo "=== FINISHED RESOLUTION ${R} ===" >> "$LOGFILE"
done
