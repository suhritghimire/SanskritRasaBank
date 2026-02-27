#!/usr/bin/env python3
"""
Extract Sanskrit verses from GRETIL Abhijñānaśākuntalam HTML.

KEY INSIGHT: In the raw HTML, every verse pāda line starts with `* ` 
at the true beginning of an HTML source line (before any <BR>).
So we parse on the RAW HTML lines (not after splitting on <BR>),
stripping only inline tags from each line.

196 distinct verse IDs // KSak_X.Y // are present.
"""

import re
import openpyxl
from openpyxl.styles import Font

# ── 1. Read raw HTML ─────────────────────────────────────
with open("/tmp/ksakunpu.htm", encoding="utf-8", errors="replace") as f:
    raw_lines = f.read().splitlines()

print(f"Total raw HTML lines: {len(raw_lines)}")

# ── Helper: strip HTML tags from a single line ──────────
def strip_tags(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&lt;","<").replace("&gt;",">").replace("&amp;","&")
    return s.strip()

# ── 2. Identify verse lines (raw lines starting with *) ──
STAR   = re.compile(r"^\*\s+(.+)$")
ID_PAT = re.compile(r"//\s*(KSak_[\d.]+[a-z]?(?:/[\d]+)?)\s*//")

records  = []
buf_raw  = []   # accumulate verse pāda texts
cur_id   = ""

for raw_line in raw_lines:
    # Check if this raw line is a verse pāda (starts with *)
    m = STAR.match(raw_line)
    if m:
        content = strip_tags(m.group(1))
        # Does this pāda carry the closing ID?
        id_m = ID_PAT.search(content)
        if id_m:
            cur_id  = id_m.group(1)
            content = ID_PAT.sub("", content).strip()
            # Also remove page refs like p.2 p.14
            content = re.sub(r"\s*p\.\s*\d+\s*", "", content).strip()
        content = content.strip(" /|")
        if content:
            buf_raw.append(content)
        if id_m:
            # Flush this complete verse
            verse = " ".join(buf_raw)
            verse = re.sub(r"\s+", " ", verse).strip()
            if len(verse) > 8:
                records.append((cur_id, verse))
            buf_raw = []
            cur_id  = ""
    else:
        # Non-verse raw line.  If we have orphan buf (verse w/o ID),
        # only flush if this is a genuine prose/section line — 
        # NOT a blank or separator line.
        cleaned = strip_tags(raw_line).strip()
        is_blank = (cleaned == "")
        is_sep   = re.match(r"^[_\-=*\s]*$", cleaned)
        if not is_blank and not is_sep and buf_raw:
            # prose line → flush incomplete verse
            verse = " ".join(buf_raw)
            verse = re.sub(r"\s+", " ", verse).strip()
            if len(verse) > 8:
                records.append((cur_id, verse))
            buf_raw = []
            cur_id  = ""

# tail flush
if buf_raw:
    verse = " ".join(buf_raw)
    if len(verse) > 8:
        records.append((cur_id, verse))

print(f"Extracted {len(records)} verse blocks")

# ── 3. Transliterate IAST → Devanagari ───────────────────
def transliterate(text):
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate as tr
        return tr(text, sanscript.IAST, sanscript.DEVANAGARI)
    except Exception:
        pass
    TABLE = [
        ("ā","आ"),("Ā","आ"),("ī","ई"),("Ī","ई"),("ū","ऊ"),("Ū","ऊ"),
        ("ṝ","ॠ"),("ṛ","ऋ"),("Ṛ","ऋ"),("ḷ","ऌ"),("ḹ","ॡ"),
        ("ai","ऐ"),("au","औ"),("e","ए"),("o","ओ"),
        ("a","अ"),("i","इ"),("u","उ"),
        ("ṃ","ं"),("ḥ","ः"),
        ("kh","ख"),("gh","घ"),("ch","छ"),("jh","झ"),
        ("ṭh","ठ"),("ḍh","ढ"),("th","थ"),("dh","ध"),
        ("ph","फ"),("bh","भ"),
        ("k","क"),("g","ग"),("ṅ","ङ"),
        ("c","च"),("j","ज"),("ñ","ञ"),
        ("ṭ","ट"),("ḍ","ड"),("ṇ","ण"),
        ("t","त"),("d","द"),("n","न"),
        ("p","प"),("b","ब"),("m","म"),
        ("y","य"),("r","र"),("l","ल"),("v","व"),
        ("ś","श"),("Ś","श"),("ṣ","ष"),("Ṣ","ष"),
        ("s","स"),("h","ह"),("ḻ","ळ"),
        ("||","॥"),("|","।"),
    ]
    out, i = [], 0
    while i < len(text):
        for src, dst in TABLE:
            if text[i:i+len(src)] == src:
                out.append(dst); i += len(src); break
        else:
            out.append(text[i]); i += 1
    return "".join(out)

print("Transliterating …")
final = [(vid, transliterate(v)) for vid, v in records]
print(f"Done — {len(final)} rows")

# ── 4. Write Excel ────────────────────────────────────────
OUT = "/Users/suhritghimire/Downloads/NavaRasaBank-main/new_dataset_gretil.xlsx"
wb  = openpyxl.Workbook()
ws  = wb.active
ws.title = "Verses"

ws.append(["sanskrit_sholka", "verse_id"])
for cell in ws[1]:
    cell.font = Font(bold=True)

for vid, dev in final:
    ws.append([dev, vid])

ws.column_dimensions["A"].width = 110
ws.column_dimensions["B"].width = 18
wb.save(OUT)

print(f"\n✅  Saved → {OUT}")
print(f"   Total rows: {len(final)}")
print("\nSample (first 6):")
for vid, dev in final[:6]:
    print(f"  [{vid}]  {dev[:95]}")
