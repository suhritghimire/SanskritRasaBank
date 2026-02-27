#!/usr/bin/env python3
"""
openai_batch_retrieve.py
=======================
Checks the status of the OpenAI Batch job.
If completed, downloads the results and merges them into FINAL_SECOND_LABELED_OPENAI.xlsx.
"""

import os
import json
import logging
import pandas as pd
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
    if not os.path.exists(BATCH_INFO_FILE):
        log.error(f"Batch info file not found: {BATCH_INFO_FILE}")
        return

    with open(BATCH_INFO_FILE, "r") as f:
        info = json.load(f)
    
    batch_id = info["batch_id"]
    output_path = info["output_path"]
    input_path = info["input_path"]
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    log.info(f"Checking status for Batch ID: {batch_id}")
    batch = client.batches.retrieve(batch_id)
    log.info(f"Status: {batch.status}")
    
    if batch.status == "completed":
        output_file_id = batch.output_file_id
        log.info(f"Batch completed! Output File ID: {output_file_id}")
        
        # Download results
        file_response = client.files.content(output_file_id)
        results_text = file_response.text
        
        # Parse results
        results = {}
        for line in results_text.splitlines():
            if not line.strip(): continue
            data = json.loads(line)
            custom_id = data.get("custom_id")
            # Extract response content
            content = data.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            results[custom_id] = normalize_rasa(content)
            
        log.info(f"Parsed {len(results)} labeled verses.")
        
        # Merge into Excel
        log.info(f"Merging results into {output_path}...")
        
        # Load the base dataframe
        # We always start from the latest Groq-labeled file to keep it centered
        INPUT_FOR_MERGE = "/Users/suhritghimire/Downloads/NavaRasaBank-main/FINAL_SECOND_LABELED.xlsx"
        df = pd.read_excel(INPUT_FOR_MERGE, engine='openpyxl')
        
        if 'openai_rasa' not in df.columns:
            df['openai_rasa'] = None

        for idx_str, rasa in results.items():
            df.at[int(idx_str), 'openai_rasa'] = rasa
            
        df.to_excel(output_path, index=False)
        log.info(f"Results successfully saved to {output_path}")
        
    elif batch.status == "failed":
        log.error("Batch job failed.")
        if batch.errors:
            log.error(f"Errors: {batch.errors}")
    else:
        log.info(f"Job is still {batch.status}. Progress: {batch.request_counts}")

if __name__ == "__main__":
    main()
