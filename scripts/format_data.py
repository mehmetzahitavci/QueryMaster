import pandas as pd
import json
import os

print("Reading raw data...")
df = pd.read_csv("data/raw/sql_create_context_raw.csv")

os.makedirs("data/processed", exist_ok=True)

def create_prompt(row):
    """
    Converts each CSV row (question, schema, answer) into a single 
    text block (prompt) for the model to read during training.
    """
    prompt_text = (
        "Write a correct and optimized SQL query that answers the question "
        "using the database schema provided below.\n\n"
        f"### Database Schema (Context):\n{row['context']}\n\n"
        f"### Question:\n{row['question']}\n\n"
        f"### SQL Query (Answer):\n{row['answer']}"
    )
    
    return {"text": prompt_text}

print("Converting data to LLM training format (JSONL)...")

formatted_data = df.apply(create_prompt, axis=1).tolist()

output_path = "data/processed/sql_finetuning_data.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for item in formatted_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nSuccess! A total of {len(formatted_data)} prompts have been saved to '{output_path}'.")

print("\n--- SAMPLE TRAINING DATA (PROMPT) ---")
print(formatted_data[0]["text"])
print("-------------------------------------")