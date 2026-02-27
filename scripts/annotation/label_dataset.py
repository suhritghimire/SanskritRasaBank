#!/usr/bin/env python3
"""
Predict top-3 rasas with confidence scores for any Sanskrit dataset.

Adds 6 columns to the output Excel:
  rasa_1, confidence_1  — top predicted rasa + score
  rasa_2, confidence_2  — second rasa + score
  rasa_3, confidence_3  — third rasa + score

Usage:
  python label_dataset.py --input_file ramayana.xlsx --text_col "sanskrit text"
  python label_dataset.py --input_file rigveda.xlsx  --text_col "verse"
"""

import os
import re
import pickle
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification
from tqdm import tqdm


def clean_text(text: str) -> str:
    """
    Clean Sanskrit/Vedic text before inference:
    1. Remove Vedic accent marks (Udatta ॑, Anudatta ॒, Svarita ᳚ etc.)
    2. Remove verse numbers at end: ॥१॥ ॥2॥ ।।1.1.2।। etc.
    3. Remove pipe characters (| and ।)
    4. Remove inline Devanagari digit markers like ॥n॥
    5. Strip extra whitespace
    """
    if not isinstance(text, str):
        return ""
    # Remove Vedic accent combining characters (U+0951–U+0954, U+1CD0–U+1CFF range)
    text = re.sub(r'[\u0951\u0952\u0953\u0954\u1CD0-\u1CFF]', '', text)
    # Remove verse numbers at end: ।।1.1.2।। or ||1.1.2||
    text = re.sub(r'[।|]{2}[\d\.\s]+[।|]{2}', '', text)
    # Remove Devanagari verse markers: ॥१॥ ॥22॥
    text = re.sub(r'॥[\u0966-\u096F\d]+॥', '', text)
    # Remove standalone pipes and dandas
    text = re.sub(r'[|।॥]+', '', text)
    # Remove leftover digits and punctuation at end
    text = re.sub(r'[\d\.\s]+$', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Label dataset with top-3 rasas + confidence scores.")
    parser.add_argument("--model_fold", type=int, default=1, choices=[1,2,3,4,5],
                        help="Which fold model to use (1–5). Default: 1")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input Excel file (e.g. ramayana.xlsx)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output Excel file. Defaults to <input>_labeled.xlsx")
    parser.add_argument("--text_col", type=str, default="sanskrit text",
                        help="Column name containing Sanskrit verses. Default: 'sanskrit text'")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Inference batch size. Default: 32")
    args = parser.parse_args()

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "saved_models", f"MuRIL_fold{args.model_fold}_best")
    mapping_path = os.path.join(base_dir, "results", "label_mapping.pkl")
    input_path = args.input_file if os.path.isabs(args.input_file) else os.path.join(base_dir, args.input_file)

    if args.output_file:
        output_path = args.output_file if os.path.isabs(args.output_file) else os.path.join(base_dir, args.output_file)
    else:
        name, ext = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(base_dir, f"{name}_labeled.xlsx")

    # Validate paths
    for path, desc in [(model_path, "Model"), (input_path, "Input file")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{desc} not found at: {path}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load label mapping
    if os.path.exists(mapping_path):
        with open(mapping_path, "rb") as f:
            mapping = pickle.load(f)
        id2label = mapping['id2label']
    else:
        # Fallback to fixed mapping if pkl not found
        id2label = {0:"Shringara",1:"Hasya",2:"Karuna",3:"Raudra",4:"Veera",5:"Bhayanaka",6:"Bibhatsa",7:"Adbhuta",8:"Shanta"}
    print(f"Label mapping: {id2label}")

    # Load model + tokenizer
    print(f"Loading MuRIL Fold {args.model_fold} from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # Load dataset
    print(f"Loading dataset: {input_path}")
    df = pd.read_excel(input_path)

    if args.text_col not in df.columns:
        print(f"\nColumn '{args.text_col}' not found!")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Use --text_col with one of: {list(df.columns)}")

    texts_raw = df[args.text_col].astype(str).fillna("").tolist()
    texts = [clean_text(t) for t in texts_raw]
    print(f"Total verses to classify: {len(texts)}")
    print(f"Sample cleaned text: {texts[0][:80]}...")

    # Inference
    all_top3_rasas = []      # list of [rasa1, rasa2, rasa3]
    all_top3_scores = []     # list of [score1, score2, score3]

    for i in tqdm(range(0, len(texts), args.batch_size), desc="Classifying"):
        batch_texts = texts[i:i + args.batch_size]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()  # (B, 9)

        for prob_row in probs:
            top3_idx = np.argsort(prob_row)[::-1][:3]     # top 3 indices
            top3_rasas  = [id2label[idx] for idx in top3_idx]
            top3_scores = [round(float(prob_row[idx]) * 100, 2) for idx in top3_idx]  # as %
            all_top3_rasas.append(top3_rasas)
            all_top3_scores.append(top3_scores)

    # Add 6 new columns
    df['rasa_1']       = [r[0] for r in all_top3_rasas]
    df['confidence_1'] = [f"{s[0]:.2f}%" for s in all_top3_scores]
    df['rasa_2']       = [r[1] for r in all_top3_rasas]
    df['confidence_2'] = [f"{s[1]:.2f}%" for s in all_top3_scores]
    df['rasa_3']       = [r[2] for r in all_top3_rasas]
    df['confidence_3'] = [f"{s[2]:.2f}%" for s in all_top3_scores]

    # Save
    print(f"\nSaving labeled dataset to: {output_path}")
    df.to_excel(output_path, index=False)

    # Summary
    print("\n=== Prediction Summary ===")
    print(f"Total verses classified : {len(df)}")
    print(f"\nTop-1 Rasa Distribution:")
    print(df['rasa_1'].value_counts().to_string())
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
