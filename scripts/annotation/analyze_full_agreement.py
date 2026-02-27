import pandas as pd
import os

MASTER_PATH = 'FINAL_SECOND_LABELED_MASTER.xlsx'

if not os.path.exists(MASTER_PATH):
    print(f"Error: {MASTER_PATH} not found.")
    exit(1)

df = pd.read_excel(MASTER_PATH, engine='openpyxl')

def analyze_pair(df, col1, col2, name1, name2):
    overlap = df[df[col1].notna() & df[col2].notna()]
    print(f"\n--- {name1} vs {name2} ---")
    print(f"Overlap Rows: {len(overlap)}")
    if len(overlap) > 0:
        agreement = overlap[overlap[col1] == overlap[col2]]
        count = len(agreement)
        percent = (count / len(overlap)) * 100
        print(f"Agreement Count: {count} ({percent:.1f}%)")
        if count > 0:
            print("Agreement Distribution:")
            print(agreement[col1].value_counts().to_string())
    else:
        print("No overlap found.")

# Distribution of OpenAI
print(f"Total OpenAI Labels: {df['openai_rasa'].notna().sum()}")
print("OpenAI Distribution:")
print(df['openai_rasa'].value_counts().to_string())

analyze_pair(df, 'groq_rasa', 'openai_rasa', 'Groq', 'OpenAI')
analyze_pair(df, 'openai_rasa', 'deepseek_rasa', 'OpenAI', 'DeepSeek')
analyze_pair(df, 'groq_rasa', 'deepseek_rasa', 'Groq', 'DeepSeek')

# Triple agreement
triple = df[df['groq_rasa'].notna() & df['openai_rasa'].notna() & df['deepseek_rasa'].notna()]
print(f"\n--- Triple Overlap (Groq, OpenAI, DeepSeek) ---")
print(f"Rows: {len(triple)}")
if len(triple) > 0:
    triple_agree = triple[(triple['groq_rasa'] == triple['openai_rasa']) & (triple['openai_rasa'] == triple['deepseek_rasa'])]
    print(f"Triple Agreement Count: {len(triple_agree)}")
    if len(triple_agree) > 0:
        print("Triple Agreement Distribution:")
        print(triple_agree['groq_rasa'].value_counts().to_string())
