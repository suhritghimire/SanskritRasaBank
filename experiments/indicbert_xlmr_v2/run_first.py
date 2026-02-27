#!/usr/bin/env python3
"""
============================================================
NAVARASA COMPARISON TRAINING — ENTRY POINT
============================================================
Trains XLM-RoBERTa-large and IndicBERT on MERGED_FINAL.xlsx
with same parameters as MuRIL v2 for fair comparison.

Usage: python run_first.py
"""

import os
import sys
import subprocess

print("=" * 60)
print("NAVARASA — XLM-R + IndicBERT COMPARISON TRAINING")
print("=" * 60)

# Check GPU
try:
    import torch
    print(f"\n PyTorch : {torch.__version__}")
    print(f" GPU     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" GPU Name: {torch.cuda.get_device_name(0)}")
        print(f" VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    print(" PyTorch not installed yet — will install now")

# Check dataset
dataset_path = os.path.join(os.getcwd(), "/teamspace/studios/this_studio/MERGED_FINAL.xlsx")
if not os.path.exists(dataset_path):
    print(f"\n Dataset not found at: {dataset_path}")
    print(" Please upload MERGED_FINAL.xlsx to the same folder as this script.")
    sys.exit(1)
else:
    print(f"\n Dataset found: MERGED_FINAL.xlsx")

print("\n Training Plan:")
print("   Models     : XLM-RoBERTa-large, IndicBERTv2")
print("   Folds      : 5-fold stratified CV")
print("   Loss       : Focal Loss (gamma=2) + Label Smoothing (0.1)")
print("   Metric     : Weighted F1 (same as MuRIL v2)")
print("   Parameters : Identical to MuRIL v2 for fair comparison")
print("\n Est. time on T4: ~8–12 hours total (4 models × 5 folds)")

# Install requirements
print("\n Installing requirements...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "-r", "code/requirements.txt"
])

# Run training
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(" Checkpoints → checkpoints/")
print(" Models      → saved_models/")
print(" Reports     → results/")
print()

subprocess.run([sys.executable, "code/02_train.py"])
