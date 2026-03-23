# QueryMaster: End-to-End Text-to-SQL Autonomous Agent

QueryMaster is an AI-driven, full-stack data engineering and microservices solution designed to autonomously translate natural language questions into complex, optimized SQL queries and visualize the results. 

Built on a robust **Microservices Architecture**, this project combines advanced **Large Language Model (LLM) fine-tuning** with a highly scalable **Java & Spring Boot backend**, completely automating the process of database querying and data visualization for non-technical users.

##  Project Architecture

The system operates on a seamless, enterprise-grade microservices pipeline:
1. **Frontend (React):** A user-friendly interface where users input natural language questions and view returned data dynamically through integrated charts (e.g., Bar, Pie, Line charts).
2. **Primary Backend (Spring Boot & Java):** The core server that handles API requests, manages business logic, securely executes generated SQL queries directly on a **PostgreSQL** database, and serves data back to the frontend.
3. **AI Inference Microservice (Python & FastAPI):** A lightweight, isolated AI server housing the fine-tuned LLM. It receives prompts from the Spring Boot backend, generates the raw SQL, and returns it for execution.
4. **AI Engine (Model Fine-Tuning):** Leveraging the state-of-the-art **Qwen3.5-9B-MLX-4bit** model, fine-tuned specifically for Text-to-SQL tasks (handling complex JOINs and subqueries) using Parameter-Efficient Fine-Tuning (LoRA) optimized for Apple Silicon (MLX).

##  Tech Stack
* **AI, Data Engineering & Inference:** Python, Pandas, Hugging Face Datasets, MLX (Apple Silicon Optimization), FastAPI.
* **Server-Side Core:** Java, Spring Boot.
* **Database:** PostgreSQL.
* **Frontend & Visualization:** React, Charting Libraries (e.g., Recharts / Chart.js).

##  Dataset & Acknowledgments
For the fine-tuning process, the model is trained on the highly comprehensive **SQaLe Text-to-SQL dataset** to ensure mastery over real-world relational schemas and advanced SQL operations.

**Citation:**
```bibtex
@inproceedings{
wolff2025sqale,
title={{SQ}aLe: A large text-to-{SQL} corpus grounded in real schemas},
author={Cornelius Wolff and Daniel Gomm and Madelon Hulsebos},
booktitle={EurIPS 2025 Workshop: AI for Tabular Data},
year={2025},
url={https://openreview.net/forum?id=6PsKDjgoEy}
} 
```
##  Development Log (Dev Diary)

### Phase 1: Data Engineering & AI Inference Setup (In Progress)
- [x] Set up an isolated Python virtual environment optimized for Apple Silicon.
- [x] Build a robust ETL pipeline to extract, shuffle, and transform a 100,000-row sample from the trl-lab/SQaLe-text-to-SQL-dataset into MLX-compatible JSONL format.
- [ ] Download and configure the Qwen3.5-9B-MLX-4bit base model.
- [ ] Execute local LoRA fine-tuning on the SQaLe dataset to master JOINs and complex aggregations.
- [ ] Wrap the fine-tuned model in a lightweight FastAPI server to act as an independent microservice.

### Phase 2: Full-Stack Integration (Upcoming)

- [ ] Set up the Java & Spring Boot project structure.
- [ ] Establish secure connections between Spring Boot and the PostgreSQL database.
- [ ] Build REST API endpoints in Spring Boot to communicate with the Python FastAPI AI microservice.
- [ ] Develop the React Frontend to capture user input, display queried tabular data, and render dynamic analytical charts.