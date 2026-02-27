#!/usr/bin/env python3
"""
openai_batch_submit_chunked.py
==============================
Generates JSONL files for the OpenAI Batch API and submits them in chunks of 5,000.
Targets unlabeled rows in FINAL_SECOND.xlsx.
"""

import os
import json
import logging
import pandas as pd
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────
BASE_PATH = "/Users/suhritghimire/Downloads/NavaRasaBank-main"
INPUT_PATH = os.path.join(BASE_PATH, "FINAL_SECOND.xlsx")
# Use the master file if it exists, otherwise use the Groq-labeled file
LATEST_MASTER = os.path.join(BASE_PATH, "FINAL_SECOND_LABELED_MASTER.xlsx")
GROQ_ONLY = os.path.join(BASE_PATH, "FINAL_SECOND_LABELED.xlsx")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHUNK_SIZE = 5000

VALID_RASAS = [
    "Shringara", "Hasya", "Karuna", "Raudra", "Veera",
    "Bhayanaka", "Bibhatsa", "Adbhuta", "Shanta"
]

SYSTEM_PROMPT = (
    "You are an expert in Sanskrit literature and Indian aesthetics (rasa theory).\n"
    "Given a Sanskrit verse, classify it into exactly ONE of the nine Navarasa categories:\n"
    f"{', '.join(VALID_RASAS)}.\n"
    "Reply with ONLY the rasa name — no explanation, no punctuation, nothing else."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

def main():
    log.info("Loading dataset...")
    df = pd.read_excel(INPUT_PATH, engine='openpyxl')
    
    # Identify what's ALREADY labeled to avoid redundant bills
    labeled_indices = set()
    
    # 1. Check if MASTER exists (merged Groq + some DeepSeek)
    if os.path.exists(LATEST_MASTER):
        df_m = pd.read_excel(LATEST_MASTER, engine='openpyxl')
        if 'openai_rasa' in df_m.columns:
            labeled_indices.update(df_m[df_m['openai_rasa'].notna()].index.tolist())
    
    # 2. Check if specific OpenAI output exists (from previous attempts)
    # Actually, we don't have results yet because we cancelled the first batch.
    
    log.info(f"Already labeled indices: {len(labeled_indices)}")

    # Prepare ALL pending requests
    all_requests = []
    for idx, row in df.iterrows():
        if idx in labeled_indices:
            continue
            
        shloka = row['shloka']
        req = {
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(shloka)}
                ],
                "max_tokens": 10,
                "temperature": 0
            }
        }
        all_requests.append(req)

    if not all_requests:
        log.info("No rows to label.")
        return

    log.info(f"Total requests to process: {len(all_requests)}")
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Split into chunks of 5,000
    batch_records = []
    for i in range(0, len(all_requests), CHUNK_SIZE):
        chunk = all_requests[i:i + CHUNK_SIZE]
        chunk_num = (i // CHUNK_SIZE) + 1
        batch_input_file = os.path.join(BASE_PATH, f"openai_batch_chunk_{chunk_num}.jsonl")
        
        log.info(f"Writing Chunk {chunk_num} ({len(chunk)} requests)...")
        with open(batch_input_file, "w", encoding="utf-8") as f:
            for req in chunk:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        
        # Upload
        with open(batch_input_file, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="batch")
        
        # Create Batch
        batch = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"chunk": str(chunk_num), "description": f"Navarasa Batch Chunk {chunk_num}"}
        )
        
        log.info(f"Batch {chunk_num} created: {batch.id}")
        batch_records.append({
            "chunk": chunk_num,
            "batch_id": batch.id,
            "file_id": uploaded_file.id,
            "status": batch.status
        })

    # Save tracking info
    with open(os.path.join(BASE_PATH, "openai_batch_records.json"), "w") as f:
        json.dump(batch_records, f, indent=4)
    
    log.info(f"Submitted {len(batch_records)} batches. Tracking info in openai_batch_records.json")

if __name__ == "__main__":
    main()
