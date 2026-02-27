#!/usr/bin/env python3
"""
============================================================
NAVARASA CLASSIFICATION — MuRIL Inference
============================================================
Loads a trained MuRIL model and provides rasa classification 
with confidence scores (probabilities).

Usage:
    python predict.py "Sanskrit verse here"
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification

# ============================================================
# CONFIGURATION
# ============================================================

# Point this to your best fold's saved model
MODEL_PATH = "saved_models/MuRIL_fold1_best" 

ID2LABEL = {
    0: "Shringara", 1: "Hasya", 2: "Karuna", 3: "Raudra",
    4: "Veera", 5: "Bhayanaka", 6: "Bibhatsa", 7: "Adbhuta", 8: "Shanta"
}

def predict_rasa(text, model, tokenizer, device):
    """Predicts Rasa and returns all class probabilities."""
    
    # Preprocess text
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=256
    ).to(device)
    
    # Get model logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Apply Softmax to get probabilities (0.0 to 1.0)
    probs = F.softmax(logits, dim=-1).squeeze()
    
    # Combine with labels and sort by confidence
    results = []
    for i, prob in enumerate(probs):
        results.append({
            "rasa": ID2LABEL[i],
            "confidence": float(prob)
        })
    
    # Sort results highest confidence first
    results = sorted(results, key=lambda x: x["confidence"], reverse=True)
    return results

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"सङ्क्रान्तिकालवत्तस्माद् द्विजातीनां ...\"")
        sys.exit(1)
        
    input_text = sys.argv[1]
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run train.py first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    
    print("\n--- INFERENCE RESULTS ---")
    print(f"Verse: {input_text}\n")
    
    predictions = predict_rasa(input_text, model, tokenizer, device)
    
    # Display Top 3 results
    for i, pred in enumerate(predictions[:3]):
        # Format the confidence as a percentage
        conf_pct = pred['confidence'] * 100
        print(f"{i+1}. {pred['rasa']:<10} : {pred['confidence']:.4f} ({conf_pct:.1f}%)")

if __name__ == "__main__":
    main()
