import random
import os

INPUT_FILE = "data/processed/sql_finetuning_data.jsonl"
OUTPUT_DIR = "data/lora_data"

print("Starting data split for MLX LoRA training...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

random.seed(42) 
random.shuffle(lines)

valid_size = 1000
train_size = len(lines) - valid_size

train_lines = lines[:train_size]
valid_lines = lines[train_size:]

with open(os.path.join(OUTPUT_DIR, 'train.jsonl'), 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

with open(os.path.join(OUTPUT_DIR, 'valid.jsonl'), 'w', encoding='utf-8') as f:
    f.writelines(valid_lines)

print(f"Success! Data split into:")
print(f" - Training set: {len(train_lines)} rows -> {OUTPUT_DIR}/train.jsonl")
print(f" - Validation set: {len(valid_lines)} rows -> {OUTPUT_DIR}/valid.jsonl")