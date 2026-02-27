#!/usr/bin/env python3
"""
label_final_groq.py
===================
Labels the FINAL_SECOND.xlsx dataset using Groq's LLaMA-3.1-8b-instant.
Includes batching, checkpointing (auto-resume), and rasa normalization.

Usage:
    python label_final_groq.py
"""

import os
import time
import logging
import pandas as pd
from tqdm import tqdm
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

VALID_RASAS = [
    "Shringara", "Hasya", "Karuna", "Raudra", "Veera",
    "Bhayanaka", "Bibhatsa", "Adbhuta", "Shanta"
]

_NORM_MAP = {
    "shantha": "Shanta", "shanta": "Shanta",
    "sringara": "Shringara", "shringara": "Shringara",
    "veera": "Veera", "karuna": "Karuna", "raudra": "Raudra",
    "bhayanaka": "Bhayanaka", "bibhatsa": "Bibhatsa",
    "adbhuta": "Adbhuta", "hasya": "Hasya",
}

SYSTEM_PROMPT = (
    "You are an expert in Sanskrit literature and Indian aesthetics (rasa theory).\n"
    "Given a Sanskrit verse, classify it into exactly ONE of the nine Navarasa categories:\n"
    f"{', '.join(VALID_RASAS)}.\n"
    "Reply with ONLY the rasa name — no explanation, no punctuation, nothing else."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

def normalize_rasa(val):
    if pd.isna(val): return None
    return _NORM_MAP.get(str(val).strip().lower(), None)

def main():
    log.info(f"Loading {INPUT_PATH}")
    df = pd.read_excel(INPUT_PATH)
    
    # Initialize labeling column if not exists
    if "groq_rasa" not in df.columns:
        df["groq_rasa"] = None
        
    client = Groq(api_key=GROQ_API_KEY)
    
    # Identify indices to process (where groq_rasa is None)
    indices_to_label = df[df["groq_rasa"].isna()].index.tolist()
    log.info(f"Rows total: {len(df)}, Rows remaining: {len(indices_to_label)}")
    
    if not indices_to_label:
        log.info("No rows left to label.")
        return

    # Checkpointing frequency
    batch_save_size = 50 
    
    try:
        for count, idx in enumerate(tqdm(indices_to_label, desc="Labeling")):
            verse = df.at[idx, "shloka"]
            
            # Simple retry logic for transients
            retries = 3
            for attempt in range(retries):
                try:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": str(verse)},
                        ],
                        max_tokens=10,
                        temperature=0,
                    )
                    raw = response.choices[0].message.content.strip()
                    df.at[idx, "groq_rasa"] = normalize_rasa(raw)
                    break
                except Exception as e:
                    if attempt == retries - 1:
                        log.warning(f"Failed idx {idx} after {retries} attempts: {e}")
                    else:
                        time.sleep(2 * (attempt + 1)) # exponential-ish backoff
            
            # Save checkpoint every X rows
            if (count + 1) % batch_save_size == 0:
                df.to_excel(OUTPUT_PATH, index=False)
                
    except KeyboardInterrupt:
        log.info("Interrupted. Saving progress...")
    finally:
        df.to_excel(OUTPUT_PATH, index=False)
        log.info(f"Saved labeling progress to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
