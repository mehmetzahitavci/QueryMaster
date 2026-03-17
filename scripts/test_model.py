from mlx_lm import load, generate

MODEL_NAME = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

print(f"Loading {MODEL_NAME} into M4 Pro Unified Memory...")
print("This will download the model weights (approx. 4.5 GB) on the first run. Please wait...\n")

model, tokenizer = load(MODEL_NAME)

question = "Write a PostgreSQL query to find the top 5 highest paid employees from the 'employees' table."

messages = [{"role": "user", "content": question}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f"Question: {question}")
print("-" * 50)
print("Model is generating the answer (Watch the MLX speed)...\n")

response = generate(model, tokenizer, prompt=formatted_prompt, verbose=True, max_tokens=150)