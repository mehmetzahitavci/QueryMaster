from transformers import AutoTokenizer

MODEL_NAME = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

print(f"Loading Tokenizer for {MODEL_NAME}...\n")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sample_text = "SELECT COUNT(*) FROM users WHERE age > 18;"
print(f"Original Text:\n{sample_text}\n")

encoded_tokens = tokenizer.encode(sample_text)
print(f"Encoded Tokens (What the Neural Network sees):\n{encoded_tokens}\n")

decoded_text = tokenizer.decode(encoded_tokens)
print(f"Decoded Text (Reconstructed):\n{decoded_text}\n")

print(f"Total number of tokens: {len(encoded_tokens)}")

