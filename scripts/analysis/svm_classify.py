#!/usr/bin/env python3
"""
============================================================
NAVARASA — SVM Baseline Classifier
============================================================
Best-practice SVM for 9-class Sanskrit Rasa classification.

Strategy:
  - Character n-gram TF-IDF (2-5 grams) — handles Sanskrit
    sandhi/compound morphology without word segmentation
  - LinearSVC with calibrated probabilities (CalibratedClassifierCV)
  - 5-fold cross-validation on MERGED_FINAL.xlsx
  - Predicts on full Ramayana with top-3 rasa + confidence

Usage:
  python svm_classify.py
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH  = os.path.join(BASE_DIR, "MERGED_FINAL.xlsx")
PREDICT_PATH = os.path.join(BASE_DIR, "extracted_data_labeled_ramayana.xlsx")
OUTPUT_PATH  = os.path.join(BASE_DIR, "ramayana_svm_labeled.xlsx")
MODEL_PATH   = os.path.join(BASE_DIR, "results", "svm_model.pkl")

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

ID2LABEL = {
    0: "Shringara", 1: "Hasya", 2: "Karuna", 3: "Raudra",
    4: "Veera", 5: "Bhayanaka", 6: "Bibhatsa", 7: "Adbhuta", 8: "Shanta"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# ============================================================
# TEXT CLEANING — same as label_dataset.py
# ============================================================

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove Vedic accent marks
    text = re.sub(r'[\u0951\u0952\u0953\u0954\u1CD0-\u1CFF]', '', text)
    # Remove verse numbers: ।।1.1.2।।
    text = re.sub(r'[।|]{2}[\d\.\s]+[।|]{2}', '', text)
    # Remove Devanagari verse markers: ॥१॥
    text = re.sub(r'॥[\u0966-\u096F\d]+॥', '', text)
    # Remove pipes and dandas
    text = re.sub(r'[|।॥]+', '', text)
    # Normalize whitespace
    return ' '.join(text.split()).strip()


# ============================================================
# LOAD & PREPARE TRAINING DATA
# ============================================================

print("=" * 60)
print("NAVARASA SVM CLASSIFIER")
print("=" * 60)
print(f"\nLoading training data: {TRAIN_PATH}")

df = pd.read_excel(TRAIN_PATH)
df = df[['sanskrit text', 'final_label']].dropna()
df['text_clean'] = df['sanskrit text'].apply(clean_text)
df['label'] = df['final_label'].map(LABEL2ID)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

X = df['text_clean'].values
y = df['label'].values

print(f"Total training samples: {len(df)}")
print("\nClass distribution:")
for rasa, count in df['final_label'].value_counts().items():
    print(f"  {rasa:<12}: {count:>5} ({count/len(df)*100:.1f}%)")


# ============================================================
# BUILD PIPELINE
# ============================================================

# Class weights for handling imbalance
class_weights_arr = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='char_wb',      # Character n-grams with word boundary
        ngram_range=(2, 5),      # 2-5 character n-grams (best for Sanskrit)
        min_df=2,                # Ignore very rare patterns
        max_features=200_000,    # Vocab size cap
        sublinear_tf=True,       # Log-scale TF (better for short texts)
    )),
    ('clf', CalibratedClassifierCV(
        LinearSVC(
            C=5.0,               # Regularization (tuned for Sanskrit)
            class_weight=class_weight_dict,
            max_iter=2000,
            dual=True,
        ),
        cv=3,                    # 3-fold calibration for probabilities
        method='sigmoid'         # Platt scaling
    ))
])


# ============================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    _, _, wf1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted', zero_division=0)
    _, _, mf1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro', zero_division=0)
    fold_results.append({'fold': fold, 'accuracy': acc, 'weighted_f1': wf1, 'macro_f1': mf1})
    print(f"  Fold {fold}: Acc={acc:.4f}  Weighted_F1={wf1:.4f}  Macro_F1={mf1:.4f}")

accs = [r['accuracy'] for r in fold_results]
wf1s = [r['weighted_f1'] for r in fold_results]
mf1s = [r['macro_f1'] for r in fold_results]
print(f"\n  Mean Accuracy   : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"  Mean Weighted F1: {np.mean(wf1s):.4f} ± {np.std(wf1s):.4f}")
print(f"  Mean Macro F1   : {np.mean(mf1s):.4f} ± {np.std(mf1s):.4f}")


# ============================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================

print("\nTraining final model on all data...")
pipeline.fit(X, y)

# Save model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(pipeline, f)
print(f"Model saved to {MODEL_PATH}")

# Full classification report
y_pred_all = pipeline.predict(X)
print("\nFull training set classification report:")
print(classification_report(y, y_pred_all,
      target_names=[ID2LABEL[i] for i in range(9)], zero_division=0))


# ============================================================
# PREDICT ON RAMAYANA
# ============================================================

print("\n" + "=" * 60)
print("PREDICTING ON RAMAYANA")
print("=" * 60)

ram = pd.read_excel(PREDICT_PATH)
print(f"Ramayana shlokas: {len(ram)}")
print(f"Columns: {list(ram.columns)}")

text_col = 'shloka_text'
texts_raw = ram[text_col].astype(str).fillna('').tolist()
texts_clean = [clean_text(t) for t in texts_raw]

# Predict labels
pred_labels = pipeline.predict(texts_clean)
# Predict probabilities (top-3)
pred_probs = pipeline.predict_proba(texts_clean)

top3_rasas  = []
top3_scores = []
for prob_row in pred_probs:
    top3_idx = np.argsort(prob_row)[::-1][:3]
    top3_rasas.append([ID2LABEL[i] for i in top3_idx])
    top3_scores.append([round(float(prob_row[i]) * 100, 2) for i in top3_idx])

# Keep original columns, drop any old predicted_rasa
out = ram.drop(columns=['predicted_rasa'], errors='ignore').copy()
out['rasa_1']       = [r[0] for r in top3_rasas]
out['confidence_1'] = [f"{s[0]:.2f}%" for s in top3_scores]
out['rasa_2']       = [r[1] for r in top3_rasas]
out['confidence_2'] = [f"{s[1]:.2f}%" for s in top3_scores]
out['rasa_3']       = [r[2] for r in top3_rasas]
out['confidence_3'] = [f"{s[2]:.2f}%" for s in top3_scores]

out.to_excel(OUTPUT_PATH, index=False)
print(f"\nOutput saved to: {OUTPUT_PATH}")

print("\nRasa distribution in Ramayana (SVM):")
print(out['rasa_1'].value_counts().to_string())

print("\n" + "=" * 60)
print(" DONE!")
print("=" * 60)
print(f"\nComparison with MuRIL (for paper):")
print(f"  SVM Mean Accuracy    : {np.mean(accs):.4f}")
print(f"  SVM Mean Weighted F1 : {np.mean(wf1s):.4f}")
print(f"  MuRIL Fold 1 Accuracy: 0.8073")
print(f"  MuRIL Fold 1 Weighted F1: 0.8081")
