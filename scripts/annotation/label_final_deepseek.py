#!/usr/bin/env python3
"""
label_final_deepseek.py
=======================
Labels shlokas in FINAL_SECOND.xlsx using DeepSeek API in parallel for speed.
Saves progress periodically to FINAL_SECOND_LABELED_DEEPSEEK.xlsx.
"""

import os
import time
import logging
import pandas as pd
from tqdm import tqdm
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CONCURRENCY = 20  # Increased for speed
MAX_RETRIES = 3

VALID_RASAS = [
    "Shringara", "Hasya", "Karuna", "Raudra", "Veera",
    "Bhayanaka", "Bibhatsa", "Adbhuta", "Shanta"
]

RASA_LIST_STR = ", ".join(VALID_RASAS)

SYSTEM_PROMPT = (
    "You are an expert in Sanskrit literature and Indian aesthetics (rasa theory).\n"
    "Given a Sanskrit verse, classify it into exactly ONE of the nine Navarasa categories:\n"
    f"{RASA_LIST_STR}.\n"
    "Reply with ONLY the rasa name — no explanation, no punctuation, nothing else."
)

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

def label_verse(client, verse, idx):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(verse)},
                ],
                max_tokens=10,
                temperature=0,
                timeout=30
            )
            raw = response.choices[0].message.content.strip()
            return idx, normalize_rasa(raw)
        except Exception as e:
            if "402" in str(e) or "Insufficient Balance" in str(e):
                log.error(f"FATAL: Insufficient Balance (402) for idx {idx}")
                return idx, "PAYMENT_REQUIRED"
            log.warning(f"Retry {attempt+1}/{MAX_RETRIES} for idx {idx}: {e}")
            time.sleep(1 * (attempt + 1))
    return idx, None

def safe_save(df_local, master_path):
    """
    Safely merges local changes into the master file on disk.
    This avoids overwriting columns updated by other scripts (OpenAI, Gemini).
    """
    try:
        if os.path.exists(master_path):
            df_disk = pd.read_excel(master_path, engine='openpyxl')
        else:
            df_disk = df_local.copy()

        # Update ONLY the deepseek_rasa column
        df_disk['deepseek_rasa'] = df_local['deepseek_rasa']
        
        df_disk.to_excel(master_path, index=False)
        log.info(f"Safely merged DeepSeek progress into {master_path}")
    except Exception as e:
        log.error(f"Safe save failed: {e}")

def main():
    log.info("Loading dataset...")
    if os.path.exists(OUTPUT_PATH):
        df = pd.read_excel(OUTPUT_PATH, engine='openpyxl')
        log.info(f"Resuming from existing output: {OUTPUT_PATH}")
    else:
        df = pd.read_excel(INPUT_PATH, engine='openpyxl')
        df["deepseek_rasa"] = None
        log.info(f"Starting fresh labeling from: {INPUT_PATH}")

    # Ensure verse_id and shloka columns exist
    if "shloka" not in df.columns:
        log.error("Column 'shloka' not found in dataset.")
        return

    # Identify indices to process (where deepseek_rasa is None or failed)
    indices_to_label = df[df["deepseek_rasa"].isna()].index.tolist()
    
    if not indices_to_label:
        log.info("No rows left to label. All set!")
        return

    log.info(f"Targeting {len(indices_to_label)} unlabeled rows with {CONCURRENCY} threads.")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
    
    # Process in parallel
    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            future_to_idx = {
                executor.submit(label_verse, client, df.at[idx, "shloka"], idx): idx 
                for idx in indices_to_label
            }
            
            pbar = tqdm(total=len(indices_to_label), desc="DeepSeek Parallel")
            
            for future in as_completed(future_to_idx):
                idx, result = future.result()
                if result == "PAYMENT_REQUIRED":
                    log.error("Stopping process due to payment error.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                df.at[idx, "deepseek_rasa"] = result
                pbar.update(1)
                
                # Intermediate save every 50 results
                if pbar.n % 50 == 0:
                    safe_save(df, OUTPUT_PATH)
                    
            pbar.close()

    except KeyboardInterrupt:
        log.info("Interrupted. Saving progress...")
    finally:
        safe_save(df, OUTPUT_PATH)
        log.info(f"Progress saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
