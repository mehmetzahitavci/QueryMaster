"""
  BIRD23-train-filtered + SynSQL-2.5M → MLX fine-tuning JSONL Pipeline
"""

import json
import os
import random
import sys
from transformers import AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────
TOKENIZER_ID = "Qwen/Qwen3-8B"

MAX_TOKENS   = 1600    # Truncates long schemas to prevent Out of Memory (OOM)
MIN_TOKENS   = 80      # Discards trivial/short queries
BIRD_MAX     = 6_000   # BIRD has ~6.6K high-quality rows, extract as much as possible
SYNSQL_MAX   = 15_000  # Target extraction from SynSQL
VALID_RATIO  = 0.03    # 3% for validation split
SEED         = 42

random.seed(SEED)
os.makedirs("data", exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)

SYSTEM_PROMPT = (
    "You are a senior PostgreSQL database administrator. "
    "Given the database schema below, generate an accurate SQL query "
    "that answers the user's question. Use standard SQL and PostgreSQL-compatible "
    "syntax (e.g., CTEs, window functions, LATERAL joins where appropriate). "
    "Return ONLY the raw SQL query, nothing else."
)


def count_tokens(messages: list[dict]) -> int:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return len(tokenizer.encode(text))


def make_messages(schema: str, question: str, sql: str) -> dict | None:
    """Common message builder for all datasets using ChatML format."""
    schema   = (schema   or "").strip()
    question = (question or "").strip()
    sql      = (sql      or "").strip()

    if not question or not sql:
        return None
    if len(sql) < 10:  # Skip meaningless queries like "SELECT 1"
        return None

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"Database Schema:\n{schema}\n\nQuestion: {question}"},
        {"role": "assistant", "content": sql},
    ]

    n = count_tokens(messages)
    if n < MIN_TOKENS or n > MAX_TOKENS:
        return None
    return {"messages": messages}


def get_field(row: dict, candidates: list[str], default: str = "") -> str:
    """Tries multiple possible field names and returns the first valid match."""
    for key in candidates:
        val = row.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return default


def inspect_first_row(name: str, row: dict):
    """Prints the first row details for debugging purposes."""
    print(f"\n  [{name}] Available Fields: {list(row.keys())}")
    for k, v in row.items():
        preview = str(v)[:120].replace("\n", "\\n") if v else "(empty)"
        print(f"    {k}: {preview}")
    print()


# BIRD23-train-filtered

def process_bird() -> list[dict]:
    print("\n" + "="*60)
    print("1/3  Processing BIRD23-train-filtered")
    print("="*60)

    from datasets import load_dataset
    try:
        ds = load_dataset("birdsql/bird23-train-filtered", split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return []

    print(f"  Total rows: {len(ds):,}")
    inspect_first_row("BIRD", ds[0])

    rows, skipped = [], 0
    for row in ds:
        db_id    = get_field(row, ["db_id"], "unknown_db")
        question = get_field(row, ["question"])
        evidence = get_field(row, ["evidence"])
        sql      = get_field(row, ["SQL", "sql", "query"])

        # Use explicit schema if available, otherwise construct from db_id + evidence
        schema = get_field(row, ["schema", "create_table", "db_schema", "DDL"])
        if not schema:
            schema = f"-- Database: {db_id}"
            if evidence:
                schema += f"\n-- Hints: {evidence}"

        result = make_messages(schema, question, sql)
        if result is None:
            skipped += 1
            continue

        rows.append(result)
        if len(rows) >= BIRD_MAX:
            break

    print(f"  Retained: {len(rows):,} | Skipped: {skipped:,}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# SynSQL-2.5M (iNeil77 Arrow version - optimized for speed)
# The original seeklhy version is JSONL and slow to download.
# Arrow/Parquet format allows lightning-fast streaming.
# ═══════════════════════════════════════════════════════════════════════════

def process_synsql() -> list[dict]:
    print("\n" + "="*60)
    print("2/3  Processing SynSQL-2.5M (iNeil77 Arrow Version)")
    print("="*60)

    from datasets import load_dataset
    ds = None

    # Attempt 1: iNeil77 Arrow version (fast streaming)
    print("  Attempting streaming via iNeil77/SynSQL-2.5M...")
    try:
        ds = load_dataset("iNeil77/SynSQL-2.5M", split="train", streaming=True)
        first = next(iter(ds))
        inspect_first_row("SynSQL (iNeil77)", first)
        print("  Connection successful!")
    except Exception as e:
        print(f"  iNeil77 failed: {e}")
        ds = None

    # Attempt 2: Original seeklhy version (fallback)
    if ds is None:
        print("  Attempting streaming via seeklhy/SynSQL-2.5M...")
        try:
            ds = load_dataset("seeklhy/SynSQL-2.5M", split="train", streaming=True)
            first = next(iter(ds))
            inspect_first_row("SynSQL (seeklhy)", first)
            print("  Connection successful!")
        except Exception as e:
            print(f"  seeklhy failed: {e}")
            ds = None

    if ds is None:
        print("  Failed to load SynSQL from any source. Skipping.")
        return []

    rows, skipped, seen = [], 0, 0

    print(f"  Target: Collecting {SYNSQL_MAX:,} complex examples...")
    print(f"  (This might take a few minutes)\n")

    for row in ds:
        seen += 1

        schema = get_field(row, [
            "db_schema", "schema", "db_details", "create_table",
            "database", "db_content", "context", "DDL"
        ])
        question = get_field(row, ["question", "nl_question", "natural_language_question"])
        sql      = get_field(row, ["SQL", "sql", "query", "sql_query"])

        # ── Complexity Filter ──
        complexity = get_field(row, ["complexity", "sql_complexity", "difficulty"]).lower()

        is_complex = any(c in complexity for c in
                         ["complex", "hard", "highly", "advanced", "extra"])

        # Downsample simple queries (take 1 in 10) to maintain diversity without diluting complexity
        if not is_complex and complexity and seen % 10 != 0:
            skipped += 1
            continue

        # If complexity flag is missing, heuristically check for JOINs/Subqueries
        if not complexity:
            sql_upper = sql.upper()
            has_join = "JOIN" in sql_upper
            has_subquery = sql_upper.count("SELECT") > 1
            if not has_join and not has_subquery and seen % 5 != 0:
                skipped += 1
                continue

        result = make_messages(schema, question, sql)
        if result is None:
            skipped += 1
            continue

        rows.append(result)

        if seen % 100_000 == 0:
            print(f"    Scanned: {seen:,} | Retained: {len(rows):,} | Skipped: {skipped:,}")

        if len(rows) >= SYNSQL_MAX:
            print(f"  Target reached ({SYNSQL_MAX:,})!")
            break

    print(f"  Total scanned: {seen:,} | Retained: {len(rows):,} | Skipped: {skipped:,}")
    return rows

# Gretel Synthetic Text-to-SQL 

def process_gretel(target: int = 5000) -> list[dict]:
    print("\n" + "="*60)
    print("3/3  Processing Gretel synthetic_text_to_sql (Supplemental)")
    print("="*60)

    from datasets import load_dataset
    try:
        ds = load_dataset("gretelai/synthetic_text_to_sql", split="train", streaming=True)
        first = next(iter(ds))
        inspect_first_row("Gretel", first)
    except Exception as e:
        print(f"  Failed to load Gretel: {e}")
        return []

    rows, skipped, seen = [], 0, 0

    for row in ds:
        seen += 1

        complexity = get_field(row, ["sql_complexity"]).lower()

        # Extract only complex query structures
        if complexity not in ("subqueries", "multiple_joins",
                              "aggregation", "window_functions"):
            skipped += 1
            continue

        schema   = get_field(row, ["sql_context", "schema", "context"])
        question = get_field(row, ["sql_prompt", "question"])
        sql      = get_field(row, ["sql", "query"])

        result = make_messages(schema, question, sql)
        if result is None:
            skipped += 1
            continue

        rows.append(result)

        if seen % 20_000 == 0:
            print(f"    Scanned: {seen:,} | Retained: {len(rows):,}")

        if len(rows) >= target:
            break

    print(f"  Retained: {len(rows):,} | Skipped: {skipped:,}")
    return rows


# Main Execution Pipeline

def save_jsonl(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    # ── 1. Process Datasets ──
    bird_rows   = process_bird()
    synsql_rows = process_synsql()

    # If SynSQL falls short, supplement heavily from Gretel
    gretel_rows = []
    total_so_far = len(bird_rows) + len(synsql_rows)
    if total_so_far < 10_000:
        shortfall = 10_000 - total_so_far
        print(f"\n  Currently at {total_so_far:,} samples. Shortfall of {shortfall:,} will be supplemented from Gretel.")
        gretel_rows = process_gretel(target=shortfall)
    else:
        # Still fetch a small high-quality subset from Gretel
        gretel_rows = process_gretel(target=3000)

    # ── 2. Merge and Shuffle ──
    all_rows = bird_rows + synsql_rows + gretel_rows
    random.shuffle(all_rows)

    if not all_rows:
        print("\n FATAL ERROR: No data was generated!")
        print("Please check your internet connection and HuggingFace access.")
        sys.exit(1)

    # ── 3. Train/Validation Split ──
    n_valid = max(100, int(len(all_rows) * VALID_RATIO))
    valid_rows = all_rows[:n_valid]
    train_rows = all_rows[n_valid:]

    # ── 4. Token Statistics ──
    sample_size = min(500, len(train_rows))
    sample_tokens = [count_tokens(r["messages"])
                     for r in random.sample(train_rows, sample_size)]

    # ── 5. Save Artifacts ──
    save_jsonl("data/train.jsonl", train_rows)
    save_jsonl("data/valid.jsonl", valid_rows)

    # ── 6. Final Report ──
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"  BIRD samples   : {len(bird_rows):,}")
    print(f"  SynSQL samples : {len(synsql_rows):,}")
    print(f"  Gretel samples : {len(gretel_rows):,}")
    print(f"  ─────────────────────────")
    print(f"  Training Set   : {len(train_rows):,}  →  data/train.jsonl")
    print(f"  Validation Set : {len(valid_rows):,}  →  data/valid.jsonl")
    print(f"  ─────────────────────────")
    print(f"  Avg Tokens     : {sum(sample_tokens)/len(sample_tokens):.0f}")
    print(f"  Min Tokens     : {min(sample_tokens)}")
    print(f"  Max Tokens     : {max(sample_tokens)}")
    print(f"  Token Limits   : {MIN_TOKENS}–{MAX_TOKENS}")
    print("="*60)

    print()
    print("Training Command (MacBook M4 Pro 24GB Unified Memory):")
    print("─"*50)
    print("python -m mlx_lm.lora \\")
    print("  --model mlx-community/Qwen3-8B-4bit \\")
    print("  --train \\")
    print("  --data ./data \\")
    print("  --batch-size 2 \\")
    print("  --lora-layers 16 \\")
    print("  --iters 1500 \\")
    print("  --adapter-path ./adapters")
    print()


if __name__ == "__main__":
    main()