# QueryMaster: End-to-End Text-to-SQL Autonomous Agent

QueryMaster is an AI-driven data engineering solution designed to autonomously translate natural language questions into complex, optimized SQL queries.

Built on a scalable architecture, this project combines advanced Large Language Model (LLM) fine-tuning with a lightweight frontend, automating the process of database querying for non-technical users.

## Project Architecture

The system operates on a seamless pipeline:
1. **Midterm Demo Interface (Streamlit):** A user-friendly, Python-based web application developed for the midterm presentation, allowing real-time schema input and natural language querying.
2. **AI Inference Engine (Model Fine-Tuning):** Leveraging the Qwen3-8B-4bit model, fine-tuned specifically for Text-to-SQL tasks (handling complex JOINs and subqueries) using Parameter-Efficient Fine-Tuning (LoRA) optimized for Apple Silicon (MLX).
3. **Future Microservices Integration:** Transitioning to a Java Spring Boot backend and React frontend for the final production deployment.

## Tech Stack
* **AI, Data Engineering & Inference:** Python, mlx-lm (Apple Silicon Optimization), Hugging Face Datasets.
* **Frontend (Midterm Prototype):** Streamlit.
* **Planned Production Stack:** Java, Spring Boot, PostgreSQL, React, FastAPI.

## Datasets & Evaluation
For the fine-tuning process, the model was trained on a robust mixture of datasets to ensure mastery over real-world relational schemas:
* **Training Data:** A combination of the **BIRD** and **SynSQL** datasets to expose the model to highly complex, cross-domain database schemas.
* **Evaluation Data:** The model was benchmarked using the **Spider 1.0** development set.
* **Midterm Benchmark Score:** Achieved a strict Execution Accuracy of **67.4%** on the Spider evaluation suite.

## Development Log (Dev Diary)

### Phase 1: Data Engineering, AI Inference & Prototyping (Completed for Midterm)
- [x] Set up an isolated Python virtual environment optimized for Apple Silicon.
- [x] Build a robust ETL pipeline to extract and transform the BIRD and SynSQL datasets into MLX-compatible ChatML format.
- [x] Download and configure the Qwen3-8B-4bit base model.
- [x] Execute local LoRA fine-tuning to master JOINs and complex aggregations.
- [x] Evaluate the model using the Spider benchmark.
- [x] Develop a live interactive web interface using Streamlit for the midterm demonstration.

### Phase 2: Full-Stack Integration (Upcoming for Final)
- [ ] Set up the Java & Spring Boot project structure.
- [ ] Establish secure connections between Spring Boot and a PostgreSQL database.
- [ ] Wrap the fine-tuned model in a FastAPI microservice.
- [ ] Build REST API endpoints in Spring Boot to communicate with the Python AI microservice.
- [ ] Develop the React frontend to capture user input and render dynamic analytical charts.