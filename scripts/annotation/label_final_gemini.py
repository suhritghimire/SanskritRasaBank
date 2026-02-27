#!/usr/bin/env python3
"""
label_final_gemini.py
=====================
Labels shlokas using Google Gemini API (Free Tier).
Uses 15 RPM limit to stay within free tier boundaries.
Saves progress to FINAL_SECOND_LABELED_MASTER.xlsx.
"""

import os
import time
import logging
import pandas as pd
from tqdm import tqdm
from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ─────────────────────────────────────────────────────────────
BASE_PATH = "/Users/suhritghimire/Downloads/NavaRasaBank-main"
MASTER_PATH = os.path.join(BASE_PATH, "FINAL_SECOND_LABELED_MASTER.xlsx")
GROQ_PATH = os.path.join(BASE_PATH, "FINAL_SECOND_LABELED.xlsx")

# API Key - Set this in your environment or replace below
GEMINI_API_KEY = "AIzaSyBruTUwLbxJ6B6IkvLjjsuc_w9EHcN1QX0" 

MODEL = "gemini-flash-latest" 
CONCURRENCY = 2  
RPM_LIMIT = 15
DELAY_BETWEEN_REQUESTS = 60 / RPM_LIMIT

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
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=f"{SYSTEM_PROMPT}\n\nVerse: {verse}"
        )
        raw = response.text.strip()
        return idx, normalize_rasa(raw)
    except Exception as e:
        log.warning(f"Gemini error for idx {idx}: {e}")
        return idx, None

def safe_save(df_local, master_path):
    """
    Safely merges local changes into the master file on disk.
    This avoids overwriting columns updated by other scripts (e.g. OpenAI).
    """
    try:
        # 1. Load the current master from disk
        if os.path.exists(master_path):
            df_disk = pd.read_excel(master_path, engine='openpyxl')
        else:
            return # Should not happen if we started with it

        # 2. Update ONLY the gemini_rasa column for where we have new data
        # We find indices where our local df has data but the disk might not
        # Or just sync the whole column if we assume we are the only one writing to 'gemini_rasa'
        df_disk['gemini_rasa'] = df_local['gemini_rasa']

        # 3. Save back
        df_disk.to_excel(master_path, index=False)
        log.info(f"Safely merged Gemini progress into {master_path}")
    except Exception as e:
        log.error(f"Safe save failed: {e}")

def main():
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not found. Please set it as an environment variable.")
        return

    log.info("Loading dataset...")
    if os.path.exists(MASTER_PATH):
        df = pd.read_excel(MASTER_PATH, engine='openpyxl')
    else:
        df = pd.read_excel(GROQ_PATH, engine='openpyxl')

    if 'gemini_rasa' not in df.columns:
        df['gemini_rasa'] = None

    indices_to_label = df[df['gemini_rasa'].isna()].index.tolist()
    if not indices_to_label:
        log.info("All rows already labeled by Gemini.")
        return

    log.info(f"Targeting {len(indices_to_label)} rows using Gemini ({MODEL}).")
    client = genai.Client(api_key=GEMINI_API_KEY)

    pbar = tqdm(total=len(indices_to_label), desc="Gemini Labeling")
    
    try:
        for idx in indices_to_label:
            verse = df.at[idx, 'shloka']
            _, result = label_verse(client, verse, idx)
            df.at[idx, 'gemini_rasa'] = result
            pbar.update(1)
            
            # Rate limit delay
            time.sleep(DELAY_BETWEEN_REQUESTS)
            
            # Intermediate save
            if pbar.n % 20 == 0:
                safe_save(df, MASTER_PATH)

    except KeyboardInterrupt:
        log.info("Interrupted. Saving...")
    finally:
        safe_save(df, MASTER_PATH)
        log.info(f"Final progress saved to {MASTER_PATH}")

if __name__ == "__main__":
    main()
