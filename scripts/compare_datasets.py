import pandas as pd
from datasets import load_dataset

print("Data sets are loading...\n")

# 1. Load our own dataset
df_ours = pd.read_json("data/processed/sql_finetuning_data.jsonl", lines=True)

# 2. Load the Gretel AI dataset (only converting the training part to pandas)
gretel_dataset = load_dataset("gretelai/synthetic_text_to_sql")
df_gretel = gretel_dataset['train'].to_pandas()

print("="*50)
print(" DATASET COMPARISON REPORT")
print("="*50)

# Query counts
print(f"Our data sets query count: {len(df_ours):,}")
print(f"Gretel AI query count: {len(df_gretel):,}\n")

# Complexity Analysis for our own dataset:
gretel_join_count = df_gretel['sql'].str.contains('JOIN', case=False, na=False).sum()
gretel_join_ratio = (gretel_join_count / len(df_gretel)) * 100

print(f"Our Dataset Complexity Analysis:")
print(f"- Percentage of queries containing JOIN: %{gretel_join_ratio:.2f}")

