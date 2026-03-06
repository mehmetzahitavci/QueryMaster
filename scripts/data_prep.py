from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

print("Data sets installing on HuggingFace. This process may take some time depending on your internet speed. ...")

dataset = load_dataset("b-mc2/sql-create-context")

df = pd.DataFrame(dataset['train'])

print("\n--- Data sets very first 5 lines ---")
print(df.head())

file_direction = "data/raw/sql_create_context_raw.csv"
df.to_csv(file_direction, index=False)

print(f"\nProcess is sucessfull ! Total{len(df)} lines of data '{file_direction}' saved the location.")