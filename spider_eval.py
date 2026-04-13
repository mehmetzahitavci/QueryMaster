"""
spider_eval.py  –  Spider 1.0 dev set 
=============================================================================
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL    = "mlx-community/Qwen3-8B-4bit"
DEFAULT_ADAPTER  = "./adapters_best"
SPIDER_DIR       = "./spider"           
MAX_TOKENS       = 256
TIMEOUT_SEC      = 30                   
# ────────────────────────────────────────────────────────────────────────────

def load_spider_dev(spider_dir: str) -> list[dict]:
    base = Path(spider_dir)
    candidates = [
        base / "dev.json",
        base / "evaluation_examples" / "examples" / "dev.json",
    ]
    dev_path = next((p for p in candidates if p.exists()), None)

    if dev_path is None:
        print("Error: dev.json not found!")
        sys.exit(1)

    with open(dev_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Spider dev: {len(data)} question uploaded ({dev_path})")
    return data


def load_tables_json(spider_dir: str) -> dict:
    base = Path(spider_dir)
    candidates = [
        base / "tables.json",
        base / "evaluation_examples" / "examples" / "tables.json",
    ]
    tables_path = next((p for p in candidates if p.exists()), None)

    if tables_path is None:
        print("Error: tables.json not found!")
        sys.exit(1)

    with open(tables_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    db_schemas = {}
    for db in tables_data:
        db_id = db["db_id"]
        ddl_parts = []
        for i, table_name in enumerate(db["table_names_original"]):
            cols = []
            for j, (tab_idx, col_name) in enumerate(db["column_names_original"]):
                if tab_idx == i:
                    col_type = db["column_types"][j] if j < len(db["column_types"]) else "TEXT"
                    cols.append(f"  {col_name} {col_type}")
            if cols:
                ddl = f"CREATE TABLE {table_name} (\n" + ",\n".join(cols) + "\n);"
                ddl_parts.append(ddl)

        fk_parts = []
        for fk in db.get("foreign_keys", []):
            if len(fk) == 2:
                col1_idx, col2_idx = fk
                if col1_idx < len(db["column_names_original"]) and col2_idx < len(db["column_names_original"]):
                    tab1, col1 = db["column_names_original"][col1_idx]
                    tab2, col2 = db["column_names_original"][col2_idx]
                    if tab1 >= 0 and tab2 >= 0:
                        t1 = db["table_names_original"][tab1]
                        t2 = db["table_names_original"][tab2]
                        fk_parts.append(f"-- {t1}.{col1} = {t2}.{col2}")

        schema_text = "\n".join(ddl_parts)
        if fk_parts:
            schema_text += "\n-- Foreign Keys:\n" + "\n".join(fk_parts)

        db_schemas[db_id] = schema_text

    print(f"Schema: {len(db_schemas)} veritabanı yüklendi ({tables_path})")
    return db_schemas


def get_db_path(spider_dir: str, db_id: str) -> str:
    base = Path(spider_dir)
    candidates = [
        base / "database" / db_id / f"{db_id}.sqlite",
        base / "evaluation_examples" / "database" / db_id / f"{db_id}.sqlite",
        base / "evaluation_examples" / "databases" / db_id / f"{db_id}.sqlite",
    ]
    db_path = next((p for p in candidates if p.exists()), None)
    return str(db_path) if db_path else str(candidates[0])


def generate_sql(model: str, adapter: str, schema: str, question: str) -> str:
    prompt = (
        f"<|im_start|>system\n"
        f"You are a database expert. Given the database schema below, generate an accurate SQLite query "
        f"that answers the user's question. Rules: 1. Return ONLY the raw SQL query. 2. NO trailing semicolon (;).<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Database Schema:\n{schema}\n\n"
        f"Question: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", model,
        "--adapter-path", adapter,
        "--max-tokens", str(MAX_TOKENS),
        "--prompt", prompt,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        print(f"  Model hatası: {e}")
        return ""

    return parse_sql_from_output(output)


def parse_sql_from_output(output: str) -> str:
    parts = output.split("==========")
    if len(parts) >= 3:
        content = parts[1].strip()
    elif len(parts) >= 2:
        content = parts[-1].strip()
    else:
        content = output.strip()

    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    sql_match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL)
    if sql_match:
        sql = sql_match.group(1).strip()
    else:
        select_match = re.search(r"(SELECT\s+.+)", content, re.DOTALL | re.IGNORECASE)
        if select_match:
            sql = select_match.group(1).strip()
            sql = sql.split("<|im_end|>")[0].strip()
            sql = sql.split("<|im_start|>")[0].strip()
        else:
            sql = content.split("<|im_end|>")[0].strip()

    return sql.rstrip(";").strip()


def execute_sql(db_path: str, sql: str) -> list | None:
    if not sql or not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception:
        return None


def results_match(pred_results: list | None, gold_results: list | None) -> bool:
    if pred_results is None or gold_results is None:
        return False

    if len(pred_results) == 0 and len(gold_results) == 0:
        return True

    try:
        pred_set = sorted([tuple(str(v) for v in row) for row in pred_results])
        gold_set = sorted([tuple(str(v) for v in row) for row in gold_results])
        return pred_set == gold_set
    except Exception:
        return False


def get_difficulty(sql: str) -> str:
    sql_upper = sql.upper()
    join_count = sql_upper.count("JOIN")
    has_subquery = sql_upper.count("SELECT") > 1
    has_group = "GROUP BY" in sql_upper
    has_having = "HAVING" in sql_upper
    has_order = "ORDER BY" in sql_upper
    has_intersect = any(kw in sql_upper for kw in ["INTERSECT", "UNION", "EXCEPT"])

    score = join_count + has_subquery * 2 + has_group + has_having + has_intersect * 2
    if score == 0:
        return "Easy"
    elif score <= 2:
        return "Medium"
    elif score <= 4:
        return "Hard"
    else:
        return "Extra Hard"


def try_load_mlx_model(model_path: str, adapter_path: str):
    try:
        from mlx_lm import load, generate as mlx_generate
        print("MLX model Python API ile yükleniyor (çok daha hızlı)...")
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        print("Model yüklendi!")
        return model, tokenizer, mlx_generate
    except ImportError:
        print("mlx_lm Python API bulunamadı, subprocess moduna düşülüyor...")
        return None, None, None
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        print("Subprocess moduna düşülüyor...")
        return None, None, None


def generate_sql_api(model, tokenizer, mlx_generate, schema: str, question: str) -> str:
    messages = [
        {"role": "system", "content": (
            "You are a database expert. Given the database schema below, generate an accurate SQLite query "
            "that answers the user's question. "
            "Rules: "
            "1. Return ONLY the raw SQL query. "
            "2. Do NOT add a trailing semicolon (;) at the end of the query."
        )},
        {"role": "user", "content": f"Database Schema:\n{schema}\n\nQuestion: {question}"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    select_match = re.search(r"(SELECT\s+.+)", response, re.DOTALL | re.IGNORECASE)
    if select_match:
        sql = select_match.group(1).strip()
        sql = sql.split("<|im_end|>")[0].strip()
    else:
        sql = response.strip()

    return sql.rstrip(";").strip()


def main():
    parser = argparse.ArgumentParser(description="Spider benchmark evaluation")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER, help="Adapter path")
    parser.add_argument("--spider-dir", default=SPIDER_DIR, help="Spider dataset directory")
    parser.add_argument("--limit", type=int, default=0, help="Test only the first N questions (0=all)")
    parser.add_argument("--output", default="spider_predictions.json", help="Prediction output file")
    parser.add_argument("--use-subprocess", action="store_true", help="Use subprocess instead of Python API")
    args = parser.parse_args()

    print("="*60)
    print("Spider 1.0 Dev Set — Execution Accuracy Evaluation")
    print("="*60)

    dev_data = load_spider_dev(args.spider_dir)
    db_schemas = load_tables_json(args.spider_dir)

    if args.limit > 0:
        dev_data = dev_data[:args.limit]
        print(f"  Limit: First {args.limit} question will be tested")

    model, tokenizer, mlx_generate = None, None, None
    if not args.use_subprocess:
        model, tokenizer, mlx_generate = try_load_mlx_model(args.model, args.adapter_path)

    use_api = model is not None
    correct, total, errors = 0, 0, 0
    difficulty_stats = {}
    predictions = []
    start_time = time.time()

    for i, item in enumerate(dev_data):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item.get("query", item.get("sql", ""))
        schema = db_schemas.get(db_id, "")
        db_path = get_db_path(args.spider_dir, db_id)

        if use_api:
            pred_sql = generate_sql_api(model, tokenizer, mlx_generate, schema, question)
        else:
            pred_sql = generate_sql(args.model, args.adapter_path, schema, question)

        gold_results = execute_sql(db_path, gold_sql)
        pred_results = execute_sql(db_path, pred_sql)
        match = results_match(pred_results, gold_results)

        if match:
            correct += 1

        if pred_results is None and pred_sql:
            errors += 1

        total += 1
        difficulty = get_difficulty(gold_sql)
        if difficulty not in difficulty_stats:
            difficulty_stats[difficulty] = {"correct": 0, "total": 0}
        difficulty_stats[difficulty]["total"] += 1
        if match:
            difficulty_stats[difficulty]["correct"] += 1

        predictions.append({
            "id": i, "db_id": db_id, "question": question,
            "gold_sql": gold_sql, "pred_sql": pred_sql,
            "match": match, "difficulty": difficulty,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(dev_data):
            elapsed = time.time() - start_time
            acc = correct / total * 100 if total > 0 else 0
            print(f"  [{i+1}/{len(dev_data)}] EX: {acc:.1f}% ({correct}/{total}) | Hatalar: {errors}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    acc = correct / total * 100 if total > 0 else 0

    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"  Execution Accuracy : {acc:.1f}% ({correct}/{total})")
    print(f"  SQL Errors      : {errors} ({errors/total*100:.1f}%)")
    for diff in ["Easy", "Medium", "Hard", "Extra Hard"]:
        if diff in difficulty_stats:
            s = difficulty_stats[diff]
            d_acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
            print(f"    {diff:12s}: {d_acc:5.1f}% ({s['correct']}/{s['total']})")
    print("="*60)

if __name__ == "__main__":
    main()