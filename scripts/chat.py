from mlx_lm import load, generate

MODEL_NAME = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
ADAPTER_PATH = "adapters"

print("Loading QueryMaster Engine and LoRA adapter, please wait...\n")
model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_PATH)

print("=" * 50)
print("🚀 Welcome to QueryMaster Terminal Chat!")
print("Type your question in English to generate the corresponding SQL query.")
print("Type 'exit' or 'quit' to close the application.")
print("=" * 50 + "\n")

# Start an infinite loop for interactive chat
while True:
    user_input = input("You (Question): ")
    
    # Check for exit commands
    if user_input.lower() in ['exit', 'quit']:
        print("Shutting down the system. Goodbye!")
        break
        
    # Skip if the user presses Enter without typing anything
    if not user_input.strip():
        continue

    # Prepare the prompt using Qwen's chat template
    messages = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate the response (verbose=False prevents printing raw output with tags)
    raw_response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=150)

    # Post-processing: Remove the <|im_end|> tag and extract clean SQL
    clean_sql = raw_response.split("<|im_end|>")[0].strip()

    print(f"\nQueryMaster:\n{clean_sql}\n")
    print("-" * 50)