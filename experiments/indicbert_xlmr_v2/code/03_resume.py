#!/usr/bin/env python3
"""
Resume training from last checkpoint.
Usage: python code/03_resume.py
"""

import os
import sys
import json
import subprocess

CHECKPOINT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training_checkpoint.json")

if not os.path.exists(CHECKPOINT_FILE):
    print("No checkpoint found. Run python run_first.py first.")
    sys.exit(1)

with open(CHECKPOINT_FILE, 'r') as f:
    state = json.load(f)

print("=" * 60)
print("RESUME TRAINING")
print("=" * 60)
print(f"Completed folds: {state['completed_folds']}")
print(f"Current fold   : {state['current_fold']}")
models_done = list(state['best_models_info'].keys())
print(f"Completed      : {models_done}")
print("\n Resuming...")

subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "02_train.py")])
