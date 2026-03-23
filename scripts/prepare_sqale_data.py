import json
import os
from datasets import load_dataset

print(" Downloading the SQaLe dataset (This may take a few minutes depending on your internet speed)...\n")

# Load the comprehensive dataset you discovered
dataset = load_dataset("trl-lab/SQaLe-text-to-SQL-dataset")

# Access the 'train' split of the dataset
train_data = dataset['train']

# SHUFFLE THE DATASET (Seed 42 ensures reproducibility for scientific testing)
print(" Shuffling the dataset to ensure maximum diversity across all database domains...")
shuffled_data = train_data.shuffle(seed=42)

total_rows = len(shuffled_data)
print(f" Dataset loaded and shuffled successfully! Total available rows: {total_rows:,}")

# Create the 'data' directory in the root folder if it doesn't exist
os.makedirs("data", exist_ok=True)
output_file = "data/sqale_train.jsonl"

print("\n Converting data to MLX format (JSONL)...")

# Using a substantial sample of 100,000 highly diverse rows
SAMPLE_SIZE = 100000 

processed_count = 0

with open(output_file, 'w', encoding='utf-8') as f:
    for i in range(SAMPLE_SIZE):
        row = shuffled_data[i]
        
        # Extract columns based on the dataset structure
        question = row.get('question', '')
        schema = row.get('schema', '')
        answer_sql = row.get('sql', row.get('query', ''))
        
        # Assign a system role to strictly enforce database administrator behavior
        system_prompt = "You are a senior database administrator. Generate an accurate SQL query to answer the user's question based on the provided database schema. Return ONLY the raw SQL query."
        
        # Merge the database schema and the user's question into a single context block
        user_content = f"Database Schema:\n{schema}\n\nQuestion: {question}"
        
        # Construct the Chat Dictionary expected by the MLX framework
        mlx_format = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer_sql}
            ]
        }
        
        # Write each dictionary as a JSON line to the output file
        f.write(json.dumps(mlx_format) + '\n')
        processed_count += 1
        
        if processed_count % 10000 == 0:
            print(f"... Processed {processed_count:,} rows.")

print(f"\n Process Complete! 100,000 rows of training data saved to: {output_file}")