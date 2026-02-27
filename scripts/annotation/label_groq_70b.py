#!/usr/bin/env python3
"""
label_groq_70b.py
=================
Re-labels shlokas using Groq's llama-3.3-70b-versatile for higher quality.
Saves results as 'groq70b_rasa' column in FINAL_SECOND_LABELED_MASTER.xlsx.
Resumes from where it left off. Uses 10 parallel threads for speed.
"""

import os
import time
import logging
import pandas as pd
from tqdm import tqdm
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"
CONCURRENCY = 10  # Groq rate limits are generous but 70B is slower
MAX_RETRIES = 3
COL = "groq70b_rasa"  # New column — keeps old groq_rasa intact

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
    if not val:
        return None
    return _NORM_MAP.get(str(val).strip().lower(), None)


def label_verse(client, verse, idx):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(verse)},
                ],
                max_tokens=10,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            return idx, normalize_rasa(raw)
        except Exception as e:
            log.warning(f"Retry {attempt+1}/{MAX_RETRIES} for idx {idx}: {e}")
            time.sleep(2 * (attempt + 1))
    return idx, None


def safe_save(df_local, master_path):
    """Merge only the groq70b_rasa column into the master file safely."""
    try:
        df_disk = pd.read_excel(master_path, engine='openpyxl')
        df_disk[COL] = df_local[COL]
        df_disk.to_excel(master_path, index=False)
        log.info(f"Safely merged {COL} into {master_path}")
    except Exception as e:
        log.error(f"Safe save failed: {e}")


def main():
    log.info(f"Loading master dataset from {MASTER_PATH}")
    df = pd.read_excel(MASTER_PATH, engine='openpyxl')

    # Add column if it doesn't exist
    if COL not in df.columns:
        df[COL] = None

    indices_to_label = df[df[COL].isna()].index.tolist()
    log.info(f"Total rows: {len(df)}, Remaining: {len(indices_to_label)}")

    if not indices_to_label:
        log.info("All rows already labeled!")
        return

    client = Groq(api_key=GROQ_API_KEY)

    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            future_to_idx = {
                executor.submit(label_verse, client, df.at[idx, "shloka"], idx): idx
                for idx in indices_to_label
            }

            pbar = tqdm(total=len(indices_to_label), desc="Groq 70B Labeling")

            for future in as_completed(future_to_idx):
                idx, result = future.result()
                df.at[idx, COL] = result
                pbar.update(1)

                if pbar.n % 100 == 0:
                    safe_save(df, MASTER_PATH)

            pbar.close()

    except KeyboardInterrupt:
        log.info("Interrupted. Saving progress...")
    finally:
        safe_save(df, MASTER_PATH)
        log.info("Done.")


if __name__ == "__main__":
    main()
