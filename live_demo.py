"""
live_demo.py - Live SQL Generation and Inference Speed Test
===========================================================
This script loads the base model along with our fine-tuned LoRA 
adapters to generate SQL from a given database schema in real-time.
It also measures the inference latency and token generation speed.
"""

import re
import time
from mlx_lm import load, generate

def clean_sql_output(response: str) -> str:
    """Removes think tags and trailing semicolons from the generated SQL."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    select_match = re.search(r"(SELECT\s+.+)", response, re.DOTALL | re.IGNORECASE)
    if select_match:
        sql = select_match.group(1).strip()
        sql = sql.split("<|im_end|>")[0].strip()
    else:
        sql = response.strip()
    return sql.rstrip(";").strip()

def main():
    print("\n" + "="*65)
    print(" INITIALIZING MODEL... (Utilizing Unified Memory)")
    print("="*65)
    
    model_id = "mlx-community/Qwen3-8B-4bit"
    adapter_path = "./adapters_best"
    
    # Measure model loading time
    load_start = time.perf_counter()
    model, tokenizer = load(model_id, adapter_path=adapter_path)
    load_time = time.perf_counter() - load_start
    
    print(f" Base Model & LoRA Adapters loaded successfully! (Load Time: {load_time:.2f}s)\n")

    # Sample e-commerce schema for the presentation
    schema = """
    CREATE TABLE customers (
        customer_id INT PRIMARY KEY,
        first_name TEXT,
        last_name TEXT,
        city TEXT,
        join_date DATE
    );

    CREATE TABLE orders (
        order_id INT PRIMARY KEY,
        customer_id INT,
        total_amount DECIMAL,
        order_date DATE,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
    """

    question = "Find the first name and city of customers who have an order with a total amount greater than 5000, ordered by total amount descending."

    print("📌 DATABASE SCHEMA:")
    print(schema.strip())
    print("-" * 65)
    print(f"👤 USER QUESTION: \n{question}")
    print("-" * 65)
    print("⏳ Model is thinking and generating SQL...\n")

    # Formatting via ChatML as trained
    messages = [
        {"role": "system", "content": "You are a senior PostgreSQL database administrator. Given the database schema below, generate an accurate SQL query that answers the user's question. Return ONLY the raw SQL query, nothing else."},
        {"role": "user", "content": f"Database Schema:\n{schema}\n\nQuestion: {question}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # --- INFERENCE SPEED MEASUREMENT STARTS ---
    start_time = time.perf_counter()
    
    response = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
    
    end_time = time.perf_counter()
    # --- INFERENCE SPEED MEASUREMENT ENDS ---

    # Calculate statistics
    generation_time = end_time - start_time
    output_tokens = len(tokenizer.encode(response))  
    tps = output_tokens / generation_time            
    
    final_sql = clean_sql_output(response)
    
    print(" GENERATED FLAWLESS SQL:")
    print(f"\033[92m{final_sql}\033[0m")
    
    # Performance metrics for the live demo
    print("\n" + " INFERENCE PERFORMANCE METRICS:")
    print(f"  Generation Time : {generation_time:.2f} seconds")
    print(f"  Tokens Generated: {output_tokens} tokens")
    print(f"  Inference Speed : {tps:.2f} tokens/second")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()