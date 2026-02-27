#!/usr/bin/env python3
"""
Append Sanskrit verses from Kathāsaritsāgara (Vetala) and Tantrākhyāyika 1
to build_gradil_dataset.xlsx.

Format: [verse_id, shloka] (Devanagari)
"""

import re
import openpyxl
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

XLSX_PATH = "/Users/suhritghimire/Downloads/NavaRasaBank-main/build_gradil_dataset.xlsx"

def strip_tags(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return s.strip()

def clean_verse(text):
    # Remove underscores, asterisks, and collapse whitespace
    text = re.sub(r"[\*_]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def translit_and_clean(text):
    dev = transliterate(text, sanscript.IAST, sanscript.DEVANAGARI)
    dev = dev.replace("||", "॥").replace("|", "।")
    dev = dev.replace("।।", "॥")
    dev = re.sub(r"\s+", " ", dev).strip()
    return dev

all_verses = []

# --- 1. Kathāsaritsāgara (soksvppu.htm) ---
print("Processing Kathāsaritsāgara (Vetala)...")
with open("/tmp/soksvppu.htm", "r", encoding="utf-8", errors="replace") as f:
    text_content = re.sub(r"<[Bb][Rr]\s*/?>", "\n", f.read())
    lines = [strip_tags(l) for l in text_content.splitlines() if strip_tags(l)]

# Pattern: // SoKss_12,8.x (Vet_y.z) //
kss_pattern = re.compile(r"//\s*SoKss_[\d.,]+\s*\((Vet_[\d.]+)\)\s*//")
buf = []
for line in lines:
    if "GRETIL" in line or "description:" in line or "long a" in line:
        continue
    match = kss_pattern.search(line)
    if match:
        v_id = "KSS_" + match.group(1).replace(".", "_")
        content = kss_pattern.sub("", line).strip()
        if content:
            buf.append(content)
        full_text = " ".join(buf)
        all_verses.append((v_id, translit_and_clean(clean_verse(full_text))))
        buf = []
    else:
        # Check if it's a verse line being continued
        # Usually, if it's not a marker, it might be prose or first half of a verse.
        # However, KSS GRETIL often puts the marker on every verse.
        # Let's assume lines without markers are prose or metadata unless they look like verse parts.
        # Actually, looking at the previous grep, even half-slokas had markers?
        # Let's check: 
        # 103:pratiṣṭhānābhidhāno 'sti deśo godāvarītaṭe // SoKss_12,8.21 (Vet_0.1) //
        # 106:prāk trivikramasenākhyaḥ khyātakīrtir abhūn nṛpaḥ // SoKss_12,8.22 (Vet_0.2) //
        # Yes, each verse (sloka) seems to have a marker.
        pass

# --- 2. Tantrākhyāyika 1 (ttrkhy1u.htm) ---
print("Processing Tantrākhyāyika 1...")
with open("/tmp/ttrkhy1u.htm", "r", encoding="utf-8", errors="replace") as f:
    full_text = f.read()
    # The verses are inside [ ]
    # Example: [ jambuko huḍuyuddhena vayaṃ cāṣāḍhabhūtinā | dūtikā tantravāyena trayo 'narthās svayaṃ kṛtaḥ ||55|| ]
    # We'll find all [ ] blocks that contain ||
    verse_blocks = re.findall(r"\[(.*?)\]", full_text, re.DOTALL)

tantra_pattern = re.compile(r"\|\|(\d+)\|\|")
for block in verse_blocks:
    clean_block = strip_tags(block).replace("\n", " ").strip()
    if "||" in clean_block:
        match = tantra_pattern.search(clean_block)
        if match:
            v_num = match.group(1)
            v_id = f"Tantra1_{int(v_num):04d}"
            # Remove the ||number|| from the text
            content = tantra_pattern.sub("", clean_block).strip()
            # Remove any leading/trailing garbage like stray brackets if any
            content = content.replace("[", "").replace("]", "").strip()
            if content:
                all_verses.append((v_id, translit_and_clean(clean_verse(content))))

# --- SAVE ---
print(f"Total narrative verses extracted: {len(all_verses)}")
wb = openpyxl.load_workbook(XLSX_PATH)
ws = wb.active

for v_id, shloka in all_verses:
    ws.append([v_id, shloka])

wb.save(XLSX_PATH)
print("Saved narrative verses.")
