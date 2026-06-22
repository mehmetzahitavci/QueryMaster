# QueryMaster: End-to-End Text-to-SQL Autonomous Agent

QueryMaster is an AI-driven data engineering solution designed to autonomously translate natural language questions into complex, optimized SQL queries.

Built on a scalable architecture, this project combines advanced Large Language Model (LLM) fine-tuning with a lightweight frontend, automating the process of database querying for non-technical users.

## Project Architecture

The system operates on a seamless pipeline:
1. **Interactive Frontend (Streamlit):** A user-friendly, Python-based web application developed for the final project, allowing real-time schema input, database uploads, and natural language querying.
2. **AI Inference Engine:** Leveraging the Qwen3-8B-4bit model, fine-tuned specifically for Text-to-SQL tasks (handling complex JOINs and subqueries) using Parameter-Efficient Fine-Tuning (LoRA) optimized for Apple Silicon (MLX).
3. **Execution & Validation System:** An integrated execution engine that runs the generated SQL against an uploaded SQLite database or synthetic data, verifying accuracy against gold standards and visualizing relationships via Mermaid ER diagrams and flowcharts.

## Tech Stack
* **AI, Data Engineering & Inference:** Python, mlx-lm (Apple Silicon Optimization), Hugging Face Datasets.
* **Frontend & Visualization:** Streamlit, Mermaid.js, Pandas.
* **Database Execution:** SQLite (in-memory & file-based).
## Datasets & Evaluation
For the fine-tuning process, the model was trained on a robust mixture of datasets to ensure mastery over real-world relational schemas:
* **Training Data:** A combination of the **BIRD** and **SynSQL** datasets to expose the model to highly complex, cross-domain database schemas.
* **Evaluation Data:** The model was benchmarked using the **Spider 1.0** development set.
* **Model Benchmark Score:** Achieved a strict Execution Accuracy of **67.4%** on the Spider evaluation suite.

## Development Log (Dev Diary)

### Phase 1: Data Engineering & AI Inference
- [x] Set up an isolated Python virtual environment optimized for Apple Silicon.
- [x] Build a robust ETL pipeline to extract and transform the BIRD and SynSQL datasets into MLX-compatible ChatML format.
- [x] Download and configure the Qwen3-8B-4bit base model.
- [x] Execute local LoRA fine-tuning to master JOINs and complex aggregations.
- [x] Evaluate the model using the Spider benchmark.

### Phase 2: Advanced Application Features (Final Project)
- [x] Develop a live interactive web interface using Streamlit.
- [x] Implement dynamic SQLite execution (uploading `.db` files or generating synthetic in-memory data).
- [x] Build a robust DDL parser to automatically generate Entity-Relationship (ER) diagrams via Mermaid.
- [x] Add query execution flowcharts to visualize the logical steps of the generated SQL.
- [x] Implement Execution Accuracy Checking to validate generated queries against Expected Gold SQL via multiset comparison.
- [x] Add data export options (CSV, JSON, and comprehensive Markdown reports).