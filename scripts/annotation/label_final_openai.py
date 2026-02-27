#!/usr/bin/env python3
"""
label_final_openai.py
=====================
Labels the FINAL_SECOND_LABELED.xlsx dataset using OpenAI's gpt-4o.
Includes batching, checkpointing (auto-resume), and rasa normalization.

Usage:
    python label_final_openai.py
"""

import os
import time
import logging
import pandas as pd
from tqdm import tqdm
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
    if not os.path.exists(INPUT_PATH):
        # Fallback to the original if Groq hasn't created the labeled file yet
        INPUT_FILE = "/Users/suhritghimire/Downloads/NavaRasaBank-main/FINAL_SECOND.xlsx"
        log.info(f"Labeled file not found, falling back to {INPUT_FILE}")
    else:
        INPUT_FILE = INPUT_PATH

    log.info(f"Loading {INPUT_FILE}")
    # Using openpyxl to avoid determining format issues
    try:
        df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    except Exception as e:
        log.error(f"Failed to load Excel: {e}")
        return
    
    # Initialize labeling column if not exists
    if "openai_rasa" not in df.columns:
        df["openai_rasa"] = None
        
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Identify indices to process (where openai_rasa is None)
    indices_to_label = df[df["openai_rasa"].isna()].index.tolist()
    log.info(f"Rows total: {len(df)}, Rows remaining for OpenAI: {len(indices_to_label)}")
    
    if not indices_to_label:
        log.info("No rows left to label for OpenAI.")
        return

    # Checkpointing frequency
    batch_save_size = 50 
    
    try:
        for count, idx in enumerate(tqdm(indices_to_label, desc="OpenAI Labeling")):
            verse = df.at[idx, "shloka"]
            
            # Simple retry logic for transients
            retries = 3
            for attempt in range(retries):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": str(verse)},
                        ],
                        max_tokens=10,
                        temperature=0,
                    )
                    raw = response.choices[0].message.content.strip()
                    df.at[idx, "openai_rasa"] = normalize_rasa(raw)
                    break
                except Exception as e:
                    if attempt == retries - 1:
                        log.warning(f"Failed idx {idx} after {retries} attempts: {e}")
                    else:
                        time.sleep(1 * (attempt + 1))
            
            # Save checkpoint every X rows
            # WARNING: Serializing to the same file as Groq might lead to races if they write simultaneously.
            # However, since they both write the entire file, the last one to write wins.
            # To be safer, we write to a separate file or handle the merging later.
            # Actually, the user asked to ADD a column. 
            # If I save to FINAL_SECOND_LABELED_OPENAI.xlsx it's safer.
            if (count + 1) % batch_save_size == 0:
                df.to_excel(OUTPUT_PATH, index=False)
                
    except KeyboardInterrupt:
        log.info("Interrupted. Saving progress...")
    finally:
        df.to_excel(OUTPUT_PATH, index=False)
        log.info(f"Saved OpenAI progress to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
