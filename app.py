import streamlit as st
import time
import re
from mlx_lm import load, generate

# Page Configuration
st.set_page_config(page_title="QueryMaster: Text-to-SQL", page_icon="🤖", layout="wide")

st.title(" QueryMaster: End-to-End Text-to-SQL Agent")
st.markdown("**FEE306 Applied Artificial Neural Networks - Midterm Project Demo**")

# ─── Model Upload (Caching) ───
# @st.cache_resource it contains in RAM.
@st.cache_resource
def load_model():
    model_id = "mlx-community/Qwen3-8B-4bit"
    adapter_path = "./adapters_best"
    return load(model_id, adapter_path=adapter_path)

with st.spinner("Model loading... (Utilizing Unified Memory)"):
    model, tokenizer = load_model()

st.success(" Model and LoRA adapters loaded successfully!")

# ─── Interface Components ───
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(" Database Schema (DDL)")
    default_schema = """CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    city TEXT
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    total_amount DECIMAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);"""
    schema = st.text_area("SQL schema:", value=default_schema, height=300)

with col2:
    st.subheader(" User Question")
    question = st.text_input("What do you want to learn from the database?", value="Find the first name and city of customers who have an order with a total amount greater than 4000, ordered by total amount descending.")

    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button(" Generate SQL", use_container_width=True)

# ─── SQL Generation and Output ───
if generate_btn:
    with st.spinner("Model is thinking and writing SQL..."):
        messages = [
            {"role": "system", "content": "You are a senior PostgreSQL database administrator. Given the database schema below, generate an accurate SQL query that answers the user's question. Return ONLY the raw SQL query, nothing else."},
            {"role": "user", "content": f"Database Schema:\n{schema}\n\nQuestion: {question}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        start_time = time.perf_counter()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
        end_time = time.perf_counter()
        
        # SQL Cleaning Process
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        select_match = re.search(r"(SELECT\s+.+)", response, re.DOTALL | re.IGNORECASE)
        if select_match:
            sql = select_match.group(1).strip().split("<|im_end|>")[0].strip()
        else:
            sql = response.strip()
        sql = sql.rstrip(";").strip()
        
        gen_time = end_time - start_time

    st.subheader(" Generated SQL:")
    st.code(sql, language="sql")
    st.caption(f" Generation Time: {gen_time:.2f} seconds")