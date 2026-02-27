#!/usr/bin/env python3
"""
openai_batch_retrieve_all.py
===========================
Polls status for all chunks in openai_batch_records.json.
Downloads and merges completed results into FINAL_SECOND_LABELED_MASTER.xlsx.
"""

import os
import json
import logging
import pandas as pd
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────
BASE_PATH = "/Users/suhritghimire/Downloads/NavaRasaBank-main"
RECORDS_FILE = os.path.join(BASE_PATH, "openai_batch_records.json")
MASTER_PATH = os.path.join(BASE_PATH, "FINAL_SECOND_LABELED_MASTER.xlsx")
GROQ_LABELED_PATH = os.path.join(BASE_PATH, "FINAL_SECOND_LABELED.xlsx")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_NORM_MAP = {
    "shantha": "Shanta", "shanta": "Shanta",
    "sringara": "Shringara", "shringara": "Shringara",
    "veera": "Veera", "karuna": "Karuna", "raudra": "Raudra",
    "bhayanaka": "Bhayanaka", "bibhatsa": "Bibhatsa",
    "adbhuta": "Adbhuta", "hasya": "Hasya",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

def normalize_rasa(val):
    if not val: return None
    return _NORM_MAP.get(str(val).strip().lower(), None)

def main():
    if not os.path.exists(RECORDS_FILE):
        log.error(f"Records file not found: {RECORDS_FILE}")
        return

    with open(RECORDS_FILE, "r") as f:
        records = json.load(f)

    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Load or initialize Master
    if os.path.exists(MASTER_PATH):
        df_master = pd.read_excel(MASTER_PATH, engine='openpyxl')
        log.info(f"Loaded existing MASTER with {len(df_master)} rows.")
    elif os.path.exists(GROQ_LABELED_PATH):
        df_master = pd.read_excel(GROQ_LABELED_PATH, engine='openpyxl')
        log.info(f"Initialized MASTER from Groq-labeled file ({len(df_master)} rows).")
    else:
        log.error("Neither MASTER nor Groq-labeled file found. Run labeling first.")
        return

    if 'openai_rasa' not in df_master.columns:
        df_master['openai_rasa'] = None

    results_merged = 0
    updated_records = []

    for record in records:
        batch_id = record['batch_id']
        chunk = record['chunk']
        current_status = record.get('status')

        if current_status == "completed_and_merged":
            updated_records.append(record)
            continue

        log.info(f"Checking Chunk {chunk} ({batch_id})...")
        try:
            batch = client.batches.retrieve(batch_id)
            log.info(f"  Status: {batch.status} ({batch.request_counts.completed}/{batch.request_counts.total})")
            
            if batch.status == "completed":
                output_file_id = batch.output_file_id
                log.info(f"  Downloading results from {output_file_id}...")
                file_response = client.files.content(output_file_id)
                
                # Parse
                chunk_results = {}
                for line in file_response.text.splitlines():
                    if not line.strip(): continue
                    data = json.loads(line)
                    custom_id = data.get("custom_id")
                    content = data.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    chunk_results[custom_id] = normalize_rasa(content)
                
                # Merge
                for idx_str, rasa in chunk_results.items():
                    df_master.at[int(idx_str), 'openai_rasa'] = rasa
                
                results_merged += len(chunk_results)
                record['status'] = "completed_and_merged"
                log.info(f"  Merged {len(chunk_results)} results from Chunk {chunk}.")
            else:
                record['status'] = batch.status

        except Exception as e:
            log.error(f"  Error processing Chunk {chunk}: {e}")
        
        updated_records.append(record)

    if results_merged > 0:
        log.info(f"Saving merged results to {MASTER_PATH}...")
        df_master.to_excel(MASTER_PATH, index=False)
        log.info("Save successful.")
    else:
        log.info("No new results to merge.")

    # Save updated records
    with open(RECORDS_FILE, "w") as f:
        json.dump(updated_records, f, indent=4)

if __name__ == "__main__":
    main()
