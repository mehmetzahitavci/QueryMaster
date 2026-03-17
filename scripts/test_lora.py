from mlx_lm import load, generate

MODEL_NAME = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
ADAPTER_PATH = "adapters" 

print("Main model and adapter are being loaded...\n")
model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_PATH)

question = "Write a SQL query to list the names of all employees who earn more than 50000 in the 'sales' department."

messages = [{"role": "user", "content": question}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f"Question: {question}")
print("-" * 50)
print("Model is trying to generate SQL...\n")

response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=150)

raw_response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=150)

clean_sql = raw_response.split("<|im_end|>")[0].strip()

print(f"Cleaned SQL:\n{clean_sql}")