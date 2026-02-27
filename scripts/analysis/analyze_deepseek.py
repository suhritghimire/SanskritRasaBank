import pandas as pd
import os

MASTER_PATH = 'FINAL_SECOND_LABELED_MASTER.xlsx'

if not os.path.exists(MASTER_PATH):
    print(f"Error: {MASTER_PATH} not found.")
    exit(1)

df = pd.read_excel(MASTER_PATH, engine='openpyxl')

# 1. DeepSeek Distribution
ds_labels = df[df['deepseek_rasa'].notna()]
print(f"Total DeepSeek Labels: {len(ds_labels)}")
print("\nDeepSeek Distribution:")
print(ds_labels['deepseek_rasa'].value_counts().to_string())

# 2. Overlap between OpenAI and DeepSeek
overlap = df[df['openai_rasa'].notna() & df['deepseek_rasa'].notna()]
print(f"\nOverlap (Both OpenAI and DeepSeek have labels): {len(overlap)}")

# 3. Agreement between OpenAI and DeepSeek
agreement = overlap[overlap['openai_rasa'] == overlap['deepseek_rasa']]
print(f"Agreement Count (OpenAI == DeepSeek): {len(agreement)}")

if len(agreement) > 0:
    print("\nAgreement Distribution:")
    print(agreement['deepseek_rasa'].value_counts().to_string())
else:
    print("\nNo agreements found between OpenAI and DeepSeek (possibly due to low overlap).")
