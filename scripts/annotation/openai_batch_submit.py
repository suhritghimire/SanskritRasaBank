#!/usr/bin/env python3
"""
openai_batch_submit.py
======================
Generates a JSONL file for the OpenAI Batch API and submits it.
Targets unlabeled rows in FINAL_SECOND.xlsx (skipping those already in FINAL_SECOND_LABELED_OPENAI.xlsx).
"""

import os
import json
import logging
import pandas as pd
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
    
    # Load already labeled rows if any
    labeled_ids = set()
    if os.path.exists(LABELED_PATH):
        try:
            df_labeled = pd.read_excel(LABELED_PATH, engine='openpyxl')
            labeled_ids = set(df_labeled[df_labeled['openai_rasa'].notna()]['verse_id'].tolist())
            log.info(f"Existing labels found for {len(labeled_ids)} verses.")
        except Exception as e:
            log.warning(f"Could not load labeled file: {e}. Starting fresh.")

    # Prepare requests
    requests = []
    count = 0
    for idx, row in df.iterrows():
        v_id = row['verse_id']
        shloka = row['shloka']
        
        if v_id in labeled_ids:
            continue
            
        # Create request structure
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
        requests.append(req)
        count += 1

    if not requests:
        log.info("No rows to label. Exiting.")
        return

    log.info(f"Writing {len(requests)} requests to {BATCH_INPUT_FILE}...")
    with open(BATCH_INPUT_FILE, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    log.info("Uploading file to OpenAI...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    with open(BATCH_INPUT_FILE, "rb") as f:
        uploaded_file = client.files.create(
            file=f,
            purpose="batch"
        )
    
    file_id = uploaded_file.id
    log.info(f"File uploaded successfully. File ID: {file_id}")

    log.info("Creating batch job...")
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Navarasa classification for remaining verses"
        }
    )
    
    batch_id = batch.id
    log.info(f"Batch created successfully. Batch ID: {batch_id}")
    log.info(f"Status: {batch.status}")

    # Save batch info
    info = {
        "batch_id": batch_id,
        "file_id": file_id,
        "input_path": INPUT_PATH,
        "output_path": LABELED_PATH,
        "created_at": batch.created_at
    }
    with open(BATCH_INFO_FILE, "w") as f:
        json.dump(info, f, indent=4)
        
    log.info(f"Batch information saved to {BATCH_INFO_FILE}")

if __name__ == "__main__":
    main()
