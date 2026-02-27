#!/usr/bin/env python3
"""
============================================================
NAVARASA INSIGHTS EXTRACTOR
============================================================
Extracts comprehensive insights from MuRIL-labeled Ramayana
and Rigveda datasets.

Outputs (saved to navarasa_insights/):
  - Rasa distribution per text (CSV + summary)
  - Per-section (Kanda / Mandala) dominant rasa breakdown
  - Top-10 highest-confidence shlokas per rasa
  - Rasa comparison table between both texts
  - Summary report (insights_summary.txt)

Usage:
  python extract_insights.py

Then download:
  tar -czf navarasa_insights.tar.gz navarasa_insights/
"""

import os
import pandas as pd
import numpy as np

OUTPUT_DIR = "navarasa_insights"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAMAYANA_FILE = "ramayana_rasa_labeled.xlsx"
RIGVEDA_FILE  = "rigveda_rasa_labeled.xlsx"

RASA_ORDER = [
    "Veera", "Karuna", "Shringara", "Raudra",
    "Adbhuta", "Bhayanaka", "Shanta", "Bibhatsa", "Hasya"
]

def load_and_validate(path, text_col, section_col, name):
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None
    df = pd.read_excel(path)
    print(f"\n{name}: {len(df)} shlokas | columns: {list(df.columns)}")
    # Parse confidence scores as floats
    for col in ['confidence_1', 'confidence_2', 'confidence_3']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%','',regex=False).astype(float)
    return df

def rasa_distribution(df, name):
    dist = df['rasa_1'].value_counts().reindex(RASA_ORDER, fill_value=0)
    dist_pct = (dist / dist.sum() * 100).round(2)
    result = pd.DataFrame({'count': dist, 'percentage': dist_pct})
    result.to_csv(os.path.join(OUTPUT_DIR, f"{name}_rasa_distribution.csv"))
    return result

def per_section_breakdown(df, section_col, name):
    if section_col not in df.columns:
        print(f"  Column '{section_col}' not found in {name}")
        return None
    breakdown = df.groupby([section_col, 'rasa_1']).size().unstack(fill_value=0)
    # Ensure all 9 rasas present
    for rasa in RASA_ORDER:
        if rasa not in breakdown.columns:
            breakdown[rasa] = 0
    breakdown = breakdown[RASA_ORDER]
    # Dominant rasa per section
    breakdown['dominant_rasa'] = breakdown[RASA_ORDER].idxmax(axis=1)
    breakdown['dominant_count'] = breakdown[RASA_ORDER].max(axis=1)
    breakdown['total_shlokas'] = breakdown[RASA_ORDER].sum(axis=1)
    breakdown['dominant_pct'] = (breakdown['dominant_count'] / breakdown['total_shlokas'] * 100).round(1)
    breakdown.to_csv(os.path.join(OUTPUT_DIR, f"{name}_section_breakdown.csv"))
    return breakdown

def top_shlokas_per_rasa(df, text_col, name, top_n=10):
    rows = []
    for rasa in RASA_ORDER:
        subset = df[df['rasa_1'] == rasa].nlargest(top_n, 'confidence_1')
        for _, row in subset.iterrows():
            rows.append({
                'rasa': rasa,
                'confidence': f"{row['confidence_1']:.2f}%",
                'text': row[text_col][:120] + '...' if len(str(row[text_col])) > 120 else row[text_col],
                'rasa_2': row.get('rasa_2', ''),
                'confidence_2': f"{row.get('confidence_2', 0):.2f}%",
            })
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUTPUT_DIR, f"{name}_top_confident_shlokas.csv"), index=False)
    return result

def write_summary(ram_dist, rig_dist, ram_sections, rig_sections):
    path = os.path.join(OUTPUT_DIR, "insights_summary.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("NAVARASA INSIGHTS — MuRIL Analysis\n")
        f.write("Ramayana & Rigveda Rasa Classification\n")
        f.write("=" * 60 + "\n\n")

        if ram_dist is not None:
            f.write("── RAMAYANA ──────────────────────────────────────────\n")
            f.write(ram_dist.to_string() + "\n\n")
            if ram_sections is not None:
                dom = ram_sections[['dominant_rasa', 'dominant_pct', 'total_shlokas']]
                f.write("Per-Kanda Dominant Rasa:\n")
                f.write(dom.to_string() + "\n\n")

        if rig_dist is not None:
            f.write("── RIGVEDA ───────────────────────────────────────────\n")
            f.write(rig_dist.to_string() + "\n\n")
            if rig_sections is not None:
                dom = rig_sections[['dominant_rasa', 'dominant_pct', 'total_shlokas']]
                f.write("Per-Mandala Dominant Rasa:\n")
                f.write(dom.to_string() + "\n\n")

        if ram_dist is not None and rig_dist is not None:
            f.write("── COMPARISON ────────────────────────────────────────\n")
            comp = pd.DataFrame({
                'Ramayana %': ram_dist['percentage'],
                'Rigveda %': rig_dist['percentage']
            }).fillna(0)
            comp['Difference'] = (comp['Ramayana %'] - comp['Rigveda %']).round(2)
            f.write(comp.to_string() + "\n\n")

        f.write("=" * 60 + "\n")
        f.write("Files saved in navarasa_insights/\n")
    print(f"\n Summary written to {path}")


def main():
    print("=" * 60)
    print("NAVARASA INSIGHTS EXTRACTOR")
    print("=" * 60)

    # Load datasets
    ram = load_and_validate(RAMAYANA_FILE, 'shloka_text', 'kanda',     'Ramayana')
    rig = load_and_validate(RIGVEDA_FILE,  'shloka_text', 'mandala',   'Rigveda')

    ram_dist, rig_dist = None, None
    ram_sections, rig_sections = None, None

    # Ramayana analysis
    if ram is not None:
        text_col  = 'shloka_text' if 'shloka_text' in ram.columns else ram.columns[1]
        sec_col   = 'kanda' if 'kanda' in ram.columns else ram.columns[0]
        print("\n[Ramayana] Rasa distribution...")
        ram_dist  = rasa_distribution(ram, "ramayana")
        print(ram_dist.to_string())
        print("\n[Ramayana] Per-Kanda breakdown...")
        ram_sections = per_section_breakdown(ram, sec_col, "ramayana")
        print("\n[Ramayana] Top confident shlokas per rasa...")
        top_shlokas_per_rasa(ram, text_col, "ramayana")

    # Rigveda analysis
    if rig is not None:
        text_col  = 'shloka_text' if 'shloka_text' in rig.columns else rig.columns[1]
        sec_col   = 'mandala' if 'mandala' in rig.columns else rig.columns[0]
        print("\n[Rigveda] Rasa distribution...")
        rig_dist  = rasa_distribution(rig, "rigveda")
        print(rig_dist.to_string())
        print("\n[Rigveda] Per-Mandala breakdown...")
        rig_sections = per_section_breakdown(rig, sec_col, "rigveda")
        print("\n[Rigveda] Top confident shlokas per rasa...")
        top_shlokas_per_rasa(rig, text_col, "rigveda")

    # Write summary
    write_summary(ram_dist, rig_dist, ram_sections, rig_sections)

    print("\n" + "=" * 60)
    print(" DONE! All insights saved to navarasa_insights/")
    print("=" * 60)
    print("\nContents:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f:<50} {size:>8} bytes")

    print("\nTo download, run:")
    print("  tar -czf navarasa_insights.tar.gz navarasa_insights/")


if __name__ == "__main__":
    main()
