# QueryMaster: End-to-End Text-to-SQL Autonomous Agent 

QueryMaster is an AI-driven, end-to-end data engineering solution designed to autonomously translate natural language questions into complex and optimized SQL queries. 

This project combines advanced **Large Language Model (LLM) fine-tuning** with a robust **Django backend architecture**, completely automating the process of database querying for non-technical users.

##  Project Architecture

The system is built on two main pillars:
1. **AI Engine (Model Fine-Tuning):** Fine-tuning an open-source LLM (e.g., Llama-3) specifically for Text-to-SQL tasks using Parameter-Efficient Fine-Tuning (LoRA) and optimized for Apple Silicon (MLX).
2. **Backend API:** A robust server-side architecture built with Django to handle API requests, interact with the fine-tuned LLM, and execute generated SQL queries securely on a PostgreSQL database.

##  Tech Stack
* **Data Engineering:** Python, Pandas, Hugging Face Datasets
* **AI & Machine Learning:** PyTorch, MLX (Apple Silicon Optimization), LoRA
* **Server-Side & API:** Django, REST Framework
* **Database:** PostgreSQL

##  Development Log (Dev Diary)

### Phase 1: Data Engineering & Pipeline setup
- [x] Set up isolated virtual environment.
- [x] Extracted `b-mc2/sql-create-context` dataset from Hugging Face.
- [x] Built a data transformation script to clean and convert raw CSV data into JSONL prompt format suitable for LLM instruction fine-tuning.

### Phase 2: Model Training (Upcoming)
- [ ] Implement base model tokenization.
- [ ] Configure LoRA adapters and MLX framework.
- [ ] Execute local fine-tuning.

### Phase 3: Backend Integration (Upcoming)
- [ ] Set up Django project and PostgreSQL database.
- [ ] Create API endpoints for natural language inputs.
- [ ] Connect the LLM inference engine to the backend.