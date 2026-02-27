#!/usr/bin/env python3
"""
============================================================
NAVARASA CLASSIFICATION — XLM-R + IndicBERT (OPTIMISED)
============================================================
Single fold, best hyperparameters for >80% accuracy.
- XLM-R: LoRA rank 32, alpha 64, LR 2e-4, batch 4
- IndicBERT: LoRA rank 32, alpha 64, LR 2e-4, batch 8
- Focal loss + label smoothing (can be adjusted)
- 15 epochs, early stopping patience 4
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
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ============================================================
# CONFIGURATION — OPTIMISED
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.environ.get(
    "SANSKRIT_DATA_PATH",
    os.path.join(BASE_DIR, "MERGED_FINAL.xlsx")
)
CHECKPOINT_FILE = os.path.join(BASE_DIR, "training_checkpoint_single.json")

for dir_name in ['saved_models', 'results', 'checkpoints', 'logs']:
    os.makedirs(os.path.join(BASE_DIR, dir_name), exist_ok=True)

MAX_LENGTH = 256
FOCAL_GAMMA = 2.0               # You can reduce to 1.5 if needed
LABEL_SMOOTHING = 0.1            # Set to 0.0 to disable

# === Fixed label mapping (9 Rasas) ===
ID2LABEL = {
    0: "Shringara", 1: "Hasya", 2: "Karuna", 3: "Raudra",
    4: "Veera", 5: "Bhayanaka", 6: "Bibhatsa", 7: "Adbhuta", 8: "Shanta"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# === Optimised Model Configs (single fold only) ===
MODELS_CONFIG = [
    {
        'name': 'XLM-RoBERTa',
        'model_name': 'xlm-roberta-large',
        'use_lora': True,
        'lora_r': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.1,
        'target_modules': ['query', 'value', 'key', 'dense'],
        'batch_size': 4,                       # Reduced to fit memory
        'gradient_accumulation_steps': 8,       # Effective batch = 32
        'learning_rate': 2e-4,                  # Higher LR for LoRA
        'epochs': 15,
        'weight_decay': 0.01,
        'warmup_ratio': 0.15,
    },
    {
        'name': 'IndicBERT',
        'model_name': 'ai4bharat/IndicBERTv2-MLM-only',
        'use_lora': True,
        'lora_r': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.1,
        'target_modules': ['query', 'value', 'key', 'dense'],
        'batch_size': 8,                        # IndicBERT is smaller
        'gradient_accumulation_steps': 4,        # Effective batch = 32
        'learning_rate': 2e-4,                   # Higher LR for LoRA
        'epochs': 15,
        'weight_decay': 0.01,
        'warmup_ratio': 0.15,
    }
]

# ============================================================
# LOSS FUNCTIONS (same as before)
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

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, focal_gamma=2.0,
                 label_smoothing=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            label_smoothing=label_smoothing
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal_loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ============================================================
# METRICS
# ============================================================

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
    return {"accuracy": acc, "weighted_f1": weighted_f1, "macro_f1": macro_f1}

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
# MAIN (single fold only)
# ============================================================

def main():
    print("=" * 60)
    print("NAVARASA — XLM-R + IndicBERT (OPTIMISED, SINGLE FOLD)")
    print("  Focal Loss | Label Smoothing | LoRA rank 32 | LR 2e-4")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    print(f" Device: {device}")

    if not os.path.exists(DATA_PATH):
        print(f" Dataset not found at: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_excel(DATA_PATH)
    df = df[['sanskrit text', 'final_label']].dropna()
    df['label'] = df['final_label'].map(LABEL2ID)

    if df['label'].isna().any():
        print("Unmapped labels found:")
        print(df[df['label'].isna()]['final_label'].unique())
        sys.exit(1)

    print(f"\nTotal samples : {len(df)}")
    print("\nClass distribution:")
    for rasa, count in df['final_label'].value_counts().items():
        print(f"  {rasa:<12}: {count:>5} ({count/len(df)*100:.1f}%)")

    with open(os.path.join(BASE_DIR, "results", "label_mapping.pkl"), "wb") as f:
        pickle.dump({'label2id': LABEL2ID, 'id2label': ID2LABEL}, f)

    # We'll run only one fold (fold 1) by manually splitting
    from sklearn.model_selection import train_test_split
    X = df['sanskrit text'].values
    y = df['label'].values

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(train_texts)} | Val: {len(val_texts)}")

    # For each model in config
    for cfg in MODELS_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Training {cfg['name']} (Fold 1)")
        print(f"{'=' * 60}")

        # Class weights
        raw_weights = compute_class_weight(
            'balanced', classes=np.unique(train_labels), y=train_labels
        )
        class_weights = torch.tensor(raw_weights, dtype=torch.float32)
        if device == "cuda":
            class_weights = class_weights.cuda()

        best_path = os.path.join(
            BASE_DIR, "saved_models", f"{cfg['name']}_fold1_best"
        )

        # Load tokenizer and base model
        print(f"Loading {cfg['name']}...")
        tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
        base_model = AutoModelForSequenceClassification.from_pretrained(
            cfg['model_name'],
            num_labels=9,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            torch_dtype=torch.float32,
            use_cache=False,
            ignore_mismatched_sizes=True
        )

        # Apply LoRA
        if cfg.get('use_lora', False):
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=cfg['lora_r'],
                lora_alpha=cfg['lora_alpha'],
                target_modules=cfg['target_modules'],
                lora_dropout=cfg['lora_dropout'],
                bias='none',
                modules_to_save=["classifier"]
            )
            model = get_peft_model(base_model, lora_config)
            model.print_trainable_parameters()
        else:
            model = base_model

        if device != "cuda":
            model = model.to(device)

        # Prepare datasets
        train_df = pd.DataFrame({'sanskrit text': train_texts, 'label': train_labels})
        val_df = pd.DataFrame({'sanskrit text': val_texts, 'label': val_labels})

        train_ds = Dataset.from_pandas(train_df).map(
            lambda x: tokenize_function(x, tokenizer), batched=True
        ).remove_columns(['sanskrit text'])

        val_ds = Dataset.from_pandas(val_df).map(
            lambda x: tokenize_function(x, tokenizer), batched=True
        ).remove_columns(['sanskrit text'])

        # Training arguments
        output_dir = os.path.join(BASE_DIR, "checkpoints", f"{cfg['name']}_fold1")
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
            logging_dir=os.path.join(BASE_DIR, "logs", f"{cfg['name']}_fold1"),
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
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
        )

        # Check for existing checkpoint (if any)
        last_checkpoint = None
        if os.path.exists(output_dir):
            ckpts = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
            if ckpts:
                last_checkpoint = os.path.join(output_dir, sorted(ckpts)[-1])
                print(f" Resuming from: {last_checkpoint}")

        print("\nStarting training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)

        print(f"\n Saving to {best_path}")
        model.save_pretrained(best_path)
        tokenizer.save_pretrained(best_path)

        # Evaluate final model
        preds = trainer.predict(val_ds)
        pred_labels = np.argmax(preds.predictions, axis=-1)

        acc = accuracy_score(val_labels, pred_labels)
        _, _, weighted_f1, _ = precision_recall_fscore_support(
            val_labels, pred_labels, average='weighted', zero_division=0
        )
        _, _, macro_f1, _ = precision_recall_fscore_support(
            val_labels, pred_labels, average='macro', zero_division=0
        )

        print(f"\n Results — {cfg['name']} (Fold 1):")
        print(f"   Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
        print(f"   Weighted F1 : {weighted_f1:.4f}")
        print(f"   Macro F1    : {macro_f1:.4f}")
        print("\n Per-class report:")
        print(classification_report(
            val_labels, pred_labels,
            target_names=list(ID2LABEL.values()), zero_division=0
        ))

        # Save report
        report_path = os.path.join(
            BASE_DIR, "results", f"{cfg['name']}_fold1_report.txt"
        )
        with open(report_path, "w") as f:
            f.write(f"{cfg['name']} — Fold 1\n")
            f.write(f"Accuracy: {acc:.4f}\nWeighted F1: {weighted_f1:.4f}\nMacro F1: {macro_f1:.4f}\n\n")
            f.write(classification_report(
                val_labels, pred_labels,
                target_names=list(ID2LABEL.values()), zero_division=0
            ))

        # Clean up
        del model, tokenizer, trainer
        if cfg.get('use_lora'):
            del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(" SINGLE FOLD TRAINING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()