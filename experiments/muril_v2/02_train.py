#!/usr/bin/env python3
"""
============================================================
NAVARASA CLASSIFICATION — MuRIL High‑Accuracy Tuning
============================================================
Optimised hyperparameters for MuRIL-large on Sanskrit Rasa:

  - Base LR        : 2e-5   (↑ from 1e-5, faster adaptation)
  - LLRD factor    : 0.9    (↓ from 0.95, stronger layer decay)
  - Focal gamma    : 1.5    (↓ from 2.0, less down‑weighting)
  - Label smoothing: 0.05   (↓ from 0.1, less regularisation)
  - Patience       : 4      (unchanged)
  - Effective batch: 32     (unchanged)

These changes aim to:
  * Allow higher‑level layers to learn more aggressively.
  * Prevent over‑smoothing that might cap performance.
  * Balance focus on hard vs. easy examples.

All other components (LLRD, focal loss, checkpointing) remain.
"""

import os
import sys
import gc
import json
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset

# ============================================================
# CONFIGURATION — OPTIMISED VALUES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.environ.get(
    "SANSKRIT_DATA_PATH",
    os.path.join(BASE_DIR, "MERGED_FINAL.xlsx")
)
CHECKPOINT_FILE = os.path.join(BASE_DIR, "training_checkpoint.json")

for dir_name in ['saved_models', 'results', 'checkpoints', 'logs']:
    os.makedirs(os.path.join(BASE_DIR, dir_name), exist_ok=True)

MAX_LENGTH = 256

# --- Tuned hyperparameters ---
FOCAL_GAMMA = 1.5          # Reduced from 2.0
LABEL_SMOOTHING = 0.05     # Reduced from 0.1
LLRD_FACTOR = 0.9          # Stronger decay (was 0.95)

ID2LABEL = {
    0: "Shringara", 1: "Hasya", 2: "Karuna", 3: "Raudra",
    4: "Veera", 5: "Bhayanaka", 6: "Bibhatsa", 7: "Adbhuta", 8: "Shanta"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

MODELS_CONFIG = [
    {
        'name': 'MuRIL',
        'model_name': 'google/muril-large-cased',
        'batch_size': 8,
        'gradient_accumulation_steps': 4,   # Effective batch = 32
        'learning_rate': 2e-5,               # Increased from 1e-5
        'epochs': 12,
        'weight_decay': 0.01,
        'warmup_ratio': 0.15,
    }
]

# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(
            logits, labels,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================
# LAYER-WISE LEARNING RATE DECAY (LLRD)
# ============================================================

def get_llrd_optimizer_params(model, base_lr, llrd_factor=0.9):
    """
    Returns parameter groups with decaying LRs.
    For muril-large (24 layers).
    """
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    
    # Classifier head gets full LR
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" in n and not any(nd in n for nd in no_decay)],
         "lr": base_lr, "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" in n and any(nd in n for nd in no_decay)],
         "lr": base_lr, "weight_decay": 0.0},
    ]
    
    num_layers = 24
    for layer_num in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (llrd_factor ** (num_layers - 1 - layer_num))
        layer_name = f"encoder.layer.{layer_num}."
        params += [
            {"params": [p for n, p in model.named_parameters()
                        if layer_name in n and not any(nd in n for nd in no_decay)],
             "lr": layer_lr, "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters()
                        if layer_name in n and any(nd in n for nd in no_decay)],
             "lr": layer_lr, "weight_decay": 0.0},
        ]
    
    # Embeddings get lowest LR
    embed_lr = base_lr * (llrd_factor ** num_layers)
    params += [
        {"params": [p for n, p in model.named_parameters()
                    if "embeddings" in n and not any(nd in n for nd in no_decay)],
         "lr": embed_lr, "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if "embeddings" in n and any(nd in n for nd in no_decay)],
         "lr": embed_lr, "weight_decay": 0.0},
    ]
    
    return [g for g in params if len(g["params"]) > 0]


# ============================================================
# CUSTOM TRAINER
# ============================================================

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, focal_gamma=2.0,
                 label_smoothing=0.1, base_lr=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_loss_fn = FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        self.base_lr = base_lr

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal_loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        opt_params = get_llrd_optimizer_params(
            self.model, self.base_lr, LLRD_FACTOR
        )
        self.optimizer = torch.optim.AdamW(
            opt_params,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        return self.optimizer


# ============================================================
# METRICS
# ============================================================

def make_compute_metrics(id2label):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        _, _, weighted_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        _, _, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        return {
            "accuracy": acc,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
        }
    return compute_metrics


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['sanskrit text'],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH
    )


# ============================================================
# CHECKPOINT MANAGER (unchanged)
# ============================================================

class CheckpointManager:
    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self.load()

    def load(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                self.state = json.load(f)
            for info in self.state.get('best_models_info', {}).values():
                if 'path' in info and not os.path.isabs(info['path']):
                    info['path'] = os.path.join(BASE_DIR, info['path'])
            print(f"\n🔄 Resuming from checkpoint")
            print(f"   Completed folds: {self.state['completed_folds']}")
        else:
            self.state = {
                'completed_folds': [],
                'completed_models_in_fold': [],
                'current_fold': 1,
                'current_model_idx': 0,
                'fold_metrics': [],
                'best_models_info': {}
            }
            print("\n🌟 Starting fresh training")

    def save(self):
        state_copy = json.loads(json.dumps(self.state))
        for info in state_copy.get('best_models_info', {}).values():
            if 'path' in info and os.path.isabs(info['path']):
                try:
                    info['path'] = os.path.relpath(info['path'], BASE_DIR)
                except ValueError:
                    pass
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state_copy, f, indent=2)

    def is_model_completed(self, fold_num, model_name):
        return f"{model_name}_fold{fold_num}" in self.state['best_models_info']

    def mark_model_completed(self, fold_num, model_name, metrics, model_path):
        key = f"{model_name}_fold{fold_num}"
        self.state['best_models_info'][key] = {
            'path': model_path,
            'accuracy': metrics['accuracy'],
            'weighted_f1': metrics['weighted_f1'],
            'macro_f1': metrics['macro_f1'],
        }
        self.state['completed_models_in_fold'].append(model_name)
        self.state['fold_metrics'].append({
            'fold': fold_num, 'model': model_name, **metrics
        })
        self.save()

    def mark_fold_completed(self, fold_num):
        if fold_num not in self.state['completed_folds']:
            self.state['completed_folds'].append(fold_num)
        self.state['completed_models_in_fold'] = []
        self.state['current_fold'] = fold_num + 1
        self.state['current_model_idx'] = 0
        self.save()

    def get_next_task(self):
        if len(self.state['completed_folds']) >= 5:
            return None, None, None
        current_fold = self.state['current_fold']
        completed_in_fold = {
            key.replace(f"_fold{current_fold}", "")
            for key in self.state['best_models_info']
            if f"_fold{current_fold}" in key
        }
        for i, cfg in enumerate(MODELS_CONFIG):
            if cfg['name'] not in completed_in_fold:
                return current_fold, i, cfg
        if len(completed_in_fold) == len(MODELS_CONFIG):
            self.mark_fold_completed(current_fold)
            return self.get_next_task()
        return None, None, None


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("NAVARASA — MuRIL High‑Accuracy Tuning (v3)")
    print("  Focal γ=1.5 | LabelSmooth=0.05 | LLRD=0.9 | LR=2e-5")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Device: {device}")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_excel(DATA_PATH)
    df = df[['sanskrit text', 'final_label']].dropna()
    df['label'] = df['final_label'].map(LABEL2ID)

    if df['label'].isna().any():
        print("⚠️ Unmapped labels found:")
        print("   ", df[df['label'].isna()]['final_label'].unique())
        sys.exit(1)

    print(f"\n📊 Total samples : {len(df)}")
    print("\n📈 Class distribution:")
    for rasa, count in df['final_label'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {rasa:<12}: {count:>5} ({pct:.1f}%)")

    with open(os.path.join(BASE_DIR, "results", "label_mapping.pkl"), "wb") as f:
        pickle.dump({'label2id': LABEL2ID, 'id2label': ID2LABEL}, f)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = df['sanskrit text'].values
    y = df['label'].values

    checkpoint = CheckpointManager(CHECKPOINT_FILE)

    while True:
        fold_num, model_idx, cfg = checkpoint.get_next_task()

        if fold_num is None:
            print("\n✅ All 5 folds complete!")
            break

        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_num}/5  |  Model: {cfg['name']}")
        print(f"{'=' * 60}")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            if fold + 1 != fold_num:
                continue

            train_texts, val_texts = X[train_idx], X[val_idx]
            train_labels, val_labels = y[train_idx], y[val_idx]
            print(f"Train: {len(train_texts)} | Val: {len(val_texts)}")

            # Class weights (balanced)
            raw_weights = compute_class_weight(
                'balanced', classes=np.unique(train_labels), y=train_labels
            )
            class_weights = torch.tensor(raw_weights, dtype=torch.float32)
            if device == "cuda":
                class_weights = class_weights.cuda()
            elif device == "mps":
                class_weights = class_weights.to("mps")

            best_path = os.path.join(
                BASE_DIR, "saved_models", f"{cfg['name']}_fold{fold_num}_best"
            )

            if checkpoint.is_model_completed(fold_num, cfg['name']):
                print("✅ Already completed — skipping")
                continue

            print("🔧 Loading tokenizer + model...")
            tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
            model = BertForSequenceClassification.from_pretrained(
                cfg['model_name'],
                num_labels=9,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                torch_dtype=torch.float32,
                ignore_mismatched_sizes=True
            )
            if device != "cuda":
                model = model.to(device)

            print("⚙️ Tokenizing...")
            train_ds = Dataset.from_pandas(
                pd.DataFrame({'sanskrit text': train_texts, 'label': train_labels})
            ).map(lambda x: tokenize_function(x, tokenizer), batched=True
            ).remove_columns(['sanskrit text'])

            val_ds = Dataset.from_pandas(
                pd.DataFrame({'sanskrit text': val_texts, 'label': val_labels})
            ).map(lambda x: tokenize_function(x, tokenizer), batched=True
            ).remove_columns(['sanskrit text'])

            output_dir = os.path.join(
                BASE_DIR, "checkpoints", f"{cfg['name']}_fold{fold_num}"
            )

            training_args = TrainingArguments(
                output_dir=output_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=cfg['learning_rate'],
                per_device_train_batch_size=cfg['batch_size'],
                per_device_eval_batch_size=cfg['batch_size'],
                num_train_epochs=cfg['epochs'],
                weight_decay=cfg['weight_decay'],
                warmup_ratio=cfg['warmup_ratio'],
                lr_scheduler_type="cosine",
                load_best_model_at_end=True,
                metric_for_best_model="eval_weighted_f1",
                greater_is_better=True,
                save_total_limit=3,
                logging_dir=os.path.join(BASE_DIR, "logs", f"{cfg['name']}_fold{fold_num}"),
                logging_steps=50,
                fp16=(device == "cuda"),
                bf16=False,
                report_to="none",
                gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
                max_grad_norm=1.0,
                dataloader_num_workers=2,
                gradient_checkpointing=True,
                optim="adamw_torch",
            )

            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=DataCollatorWithPadding(tokenizer, padding='longest', max_length=MAX_LENGTH),
                class_weights=class_weights,
                focal_gamma=FOCAL_GAMMA,
                label_smoothing=LABEL_SMOOTHING,
                base_lr=cfg['learning_rate'],
                compute_metrics=make_compute_metrics(ID2LABEL),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
            )

            last_checkpoint = None
            if os.path.exists(output_dir):
                ckpts = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
                if ckpts:
                    last_checkpoint = os.path.join(output_dir, sorted(ckpts)[-1])
                    print(f"📌 Resuming from: {last_checkpoint}")

            print("\n🚀 Starting training...")
            trainer.train(resume_from_checkpoint=last_checkpoint)

            # Save best model
            print(f"\n💾 Saving to {best_path}")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

            # Final evaluation
            preds = trainer.predict(val_ds)
            pred_labels = np.argmax(preds.predictions, axis=-1)

            acc = accuracy_score(val_labels, pred_labels)
            _, _, weighted_f1, _ = precision_recall_fscore_support(
                val_labels, pred_labels, average='weighted', zero_division=0
            )
            _, _, macro_f1, _ = precision_recall_fscore_support(
                val_labels, pred_labels, average='macro', zero_division=0
            )

            print(f"\n📊 Results — MuRIL Fold {fold_num}:")
            print(f"   Accuracy     : {acc:.4f} ({acc*100:.2f}%)")
            print(f"   Weighted F1  : {weighted_f1:.4f}")
            print(f"   Macro F1     : {macro_f1:.4f}")
            print("\n📋 Per-class report:")
            print(classification_report(
                val_labels, pred_labels,
                target_names=list(ID2LABEL.values()),
                zero_division=0
            ))

            report_path = os.path.join(BASE_DIR, "results", f"fold{fold_num}_report.txt")
            with open(report_path, "w") as f:
                f.write(f"Fold {fold_num} Results\n")
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"Weighted F1: {weighted_f1:.4f}\n")
                f.write(f"Macro F1: {macro_f1:.4f}\n\n")
                f.write(classification_report(
                    val_labels, pred_labels,
                    target_names=list(ID2LABEL.values()),
                    zero_division=0
                ))

            checkpoint.mark_model_completed(
                fold_num, cfg['name'],
                {'accuracy': acc, 'weighted_f1': weighted_f1, 'macro_f1': macro_f1},
                best_path
            )

            del model, tokenizer, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"\n✅ COMPLETED: MuRIL fold {fold_num}")
            break

    # Final summary
    print("\n" + "=" * 60)
    print("🏁 TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    ckpt = json.load(open(CHECKPOINT_FILE))
    accs = [m['accuracy'] for m in ckpt['fold_metrics']]
    wf1s = [m['weighted_f1'] for m in ckpt['fold_metrics']]
    mf1s = [m['macro_f1'] for m in ckpt['fold_metrics']]
    print(f"\n{'Fold':<6} {'Accuracy':>10} {'Weighted F1':>13} {'Macro F1':>10}")
    for m in ckpt['fold_metrics']:
        print(f"  {m['fold']:<4} {m['accuracy']:>10.4f} {m['weighted_f1']:>13.4f} {m['macro_f1']:>10.4f}")
    print(f"\n📊 Mean Accuracy   : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"📊 Mean Weighted F1: {np.mean(wf1s):.4f} ± {np.std(wf1s):.4f}")
    print(f"📊 Mean Macro F1   : {np.mean(mf1s):.4f} ± {np.std(mf1s):.4f}")
    print(f"\n📁 Reports → {BASE_DIR}/results/")
    print(f"📁 Models  → {BASE_DIR}/saved_models/")


if __name__ == "__main__":
    main()