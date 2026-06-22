import streamlit as st
import time
import re
import json
import io
import os
import random
import sqlite3
import math
import hashlib
from collections import OrderedDict, Counter
import pandas as pd

# macOS fork safety for MLX + Streamlit multiprocessing
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

from mlx_lm import load, generate

# Page Configuration
st.set_page_config(
    page_title="QueryMaster: Text-to-SQL",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  CSS Injection
def inject_css():
    """Inject CSS using Streamlit native CSS variables — no theme mismatch possible."""
    st.markdown("""
<style>
    /* 1. Hide only the "Deploy" button — keep the rest of the header bar
       (including the hamburger menu, where theme switching lives) intact. */
    [data-testid="stAppDeployButton"] {
        display: none !important;
    }

    /* 2. Reset top padding for main container */
    .block-container {
        padding-top: 1.5rem !important; 
    }

    /* 3. Push sidebar content to the top */
    [data-testid="stSidebarUserContent"] {
        padding-top: 0.5rem !important;
    }

    .qm-badge {
        display:inline-block;padding:2px 10px;border-radius:20px;
        font-size:0.73rem;font-weight:600;margin:1px 3px 1px 0;letter-spacing:0.01em;
    }
    .qm-badge-pk  { background: rgba(245,158,11,0.15);  color: #d97706; }
    .qm-badge-fk  { background: rgba(59,130,246,0.15);  color: #3b82f6; }
    .qm-badge-txt { background: var(--secondary-background-color); color: var(--text-color); opacity: 0.8; }
    .qm-badge-int { background: rgba(34,197,94,0.15);   color: #16a34a; }
    .qm-badge-dec { background: rgba(234,88,12,0.15);   color: #ea580c; }
    .qm-badge-date{ background: rgba(168,85,247,0.15);  color: #9333ea; }
    .qm-badge-bool{ background: rgba(20,184,166,0.15);  color: #0d9488; }

    .qm-stats { display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem; }
    .qm-stat { background: var(--secondary-background-color); border-radius:10px;padding:0.5rem 1rem;font-size:0.82rem; }
    .qm-stat-value { font-weight:700;font-size:1.1rem; }

    .qm-table-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius:12px;padding:0.9rem 1.1rem;margin:0.4rem;
        min-width:200px;flex:1 1 240px;transition:all 0.2s ease;
    }
    .qm-table-card:hover { border-color:rgba(100,180,255,0.4);box-shadow:0 2px 16px rgba(0,0,0,0.1); }
    .qm-table-name { font-weight:800;font-size:1rem;margin-bottom:0.5rem;padding-bottom:0.4rem;border-bottom:1px solid rgba(128,128,128,0.2); }
    .qm-col-row { font-size:0.78rem;padding:2px 0;display:flex;align-items:center;gap:4px; }
    .qm-col-type { color: var(--text-color); opacity: 0.6; font-size:0.7rem; margin-left:auto; }

    .stButton > button { border-radius:10px !important;font-weight:600 !important;transition:all 0.2s ease !important; }
    .stButton > button:hover { transform:translateY(-1px);box-shadow:0 4px 16px rgba(0,0,0,0.1); }

    [data-testid="stCodeBlock"], [data-testid="stExpander"] { border-radius:10px !important; }

    .qm-fk-arrow { color: #3b82f6; font-weight:600;font-size:0.85rem; }
</style>
    """, unsafe_allow_html=True)


# DDL Parser 
def parse_ddl(ddl_text: str) -> dict:
    """
    Parses CREATE TABLE statements and returns structured schema.
    Returns: { table_name: { columns: [{name, type, is_pk, is_fk, fk_ref}], fks: [...] } }
    """
    schema = OrderedDict()
    if not ddl_text or not ddl_text.strip():
        return schema

    ddl_text = re.sub(r'--.*$', '', ddl_text, flags=re.MULTILINE)
    ddl_text = re.sub(r'/\*.*?\*/', '', ddl_text, flags=re.DOTALL)

    # Normalize quotes and brackets
    ddl_text = ddl_text.replace('"', '').replace('`', '').replace('[', '').replace(']', '')

    # Split into individual CREATE TABLE statements
    pattern = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*?)\)\s*;',
        re.DOTALL | re.IGNORECASE
    )
    matches = pattern.findall(ddl_text)

    for table_name, body in matches:
        table_name = table_name.strip()
        columns = []
        fks = []

        # Split body by commas not inside parentheses
        lines = _split_ddl_body(body)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for inline FK constraint
            fk_match = re.match(
                r'FOREIGN\s+KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\s*\((\w+)\)',
                line, re.IGNORECASE
            )
            if fk_match:
                fks.append({
                    "column": fk_match.group(1),
                    "ref_table": fk_match.group(2),
                    "ref_column": fk_match.group(3),
                })
                # Mark the column as FK if it already exists
                for col in columns:
                    if col["name"].lower() == fk_match.group(1).lower():
                        col["is_fk"] = True
                        col["fk_ref"] = f"{fk_match.group(2)}.{fk_match.group(3)}"
                continue

            # Check for CONSTRAINT ... FOREIGN KEY
            constraint_match = re.match(
                r'CONSTRAINT\s+\w+\s+FOREIGN\s+KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\s*\((\w+)\)',
                line, re.IGNORECASE
            )
            if constraint_match:
                fks.append({
                    "column": constraint_match.group(1),
                    "ref_table": constraint_match.group(2),
                    "ref_column": constraint_match.group(3),
                })
                continue

            # Check for PRIMARY KEY on single line
            pk_single = re.search(r'PRIMARY\s+KEY\s*\((\w+)\)', line, re.IGNORECASE)
            if pk_single:
                col_name = pk_single.group(1)
                for col in columns:
                    if col["name"].lower() == col_name.lower():
                        col["is_pk"] = True
                continue

            # Check for multi-column PK
            pk_multi = re.search(r'PRIMARY\s+KEY\s*\(([\w\s,]+)\)', line, re.IGNORECASE)
            if pk_multi:
                col_names = [c.strip() for c in pk_multi.group(1).split(",")]
                for cn in col_names:
                    for col in columns:
                        if col["name"].lower() == cn.lower():
                            col["is_pk"] = True
                continue

            # Skip standalone table-level constraints we don't model as columns
            if re.match(r'(UNIQUE|CHECK|INDEX|KEY)\s*\(', line, re.IGNORECASE):
                continue
            if re.match(r'CONSTRAINT\s+\w+\s+(UNIQUE|CHECK)\s*\(', line, re.IGNORECASE):
                continue

            # Normalize internal whitespace
            line_norm = re.sub(r'\s+', ' ', line).strip()

            # Column definition
            col_match = re.match(
                r'(\w+)\s+(\w+(?:\s+\w+)?)',
                line_norm
            )
            if col_match:
                col_name = col_match.group(1)
                col_type = col_match.group(2).upper()

                _type_stopwords = {"NOT", "NULL", "UNIQUE", "DEFAULT", "PRIMARY",
                                   "REFERENCES", "CHECK", "COLLATE", "CONSTRAINT"}
                type_words = col_type.split()
                if len(type_words) > 1 and type_words[1] in _type_stopwords:
                    col_type = type_words[0]

                is_pk = "PRIMARY KEY" in line_norm.upper() or "SERIAL" in col_type
                is_fk = False
                fk_ref = None

                # Check inline REFERENCES
                ref_match = re.search(
                    r'REFERENCES\s+(\w+)\s*\((\w+)\)',
                    line_norm, re.IGNORECASE
                )
                if ref_match:
                    is_fk = True
                    fk_ref = f"{ref_match.group(1)}.{ref_match.group(2)}"
                    fks.append({
                        "column": col_name,
                        "ref_table": ref_match.group(1),
                        "ref_column": ref_match.group(2),
                    })

                columns.append({
                    "name": col_name,
                    "type": col_type.upper(),
                    "is_pk": is_pk,
                    "is_fk": is_fk,
                    "fk_ref": fk_ref,
                })

        schema[table_name] = {"columns": columns, "fks": fks}

    return schema


def _split_ddl_body(body: str) -> list:
    """Splits DDL body by top-level commas (not inside parentheses)."""
    parts = []
    depth = 0
    current = []
    for ch in body:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts

# Sample Data Generator
_FIRST_NAMES = ["John", "Jane", "Alice", "Bob", "Charlie", "Diana", "Edward",
                "Fiona", "George", "Hannah", "Ivan", "Julia", "Kevin", "Laura",
                "Michael", "Nora", "Oliver", "Paula", "Quinn", "Rachel"]
_LAST_NAMES  = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                "Miller", "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor"]
_CITIES      = ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney",
                "Toronto", "Dubai", "Singapore", "Istanbul", "Mumbai", "Seoul"]
_STATUSES    = ["active", "inactive", "pending", "completed", "cancelled"]
_GENDERS     = ["Male", "Female", "Non-binary"]
_INDUSTRIES  = ["Technology", "Finance", "Healthcare", "Education", "Retail",
                "Manufacturing", "Energy", "Transportation"]
_CATEGORIES  = ["Electronics", "Clothing", "Food", "Books", "Sports", "Music"]

def _col_type_category(col_type: str) -> str:
    t = col_type.upper()
    if any(k in t for k in ["INT", "SERIAL", "BIGINT", "SMALLINT", "TINYINT", "NUMBER"]):
        return "INT"
    if any(k in t for k in ["DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL", "MONEY"]):
        return "DECIMAL"
    if any(k in t for k in ["DATE", "TIME", "TIMESTAMP", "DATETIME"]):
        return "DATE"
    if "BOOL" in t:
        return "BOOL"
    return "TEXT"


def _generate_value(col_name: str, col_type: str, row_idx: int,
                    fk_pool: dict = None) -> any:
    cat = _col_type_category(col_type)
    name_lower = col_name.lower().replace("_", " ")

    # FK lookup
    if fk_pool and col_name in fk_pool:
        pool = fk_pool[col_name]
        if pool:
            return random.choice(pool)

    # Heuristic name-based
    if "first name" in name_lower or name_lower == "firstname":
        return random.choice(_FIRST_NAMES)
    if "last name" in name_lower or name_lower == "lastname" or "surname" in name_lower:
        return random.choice(_LAST_NAMES)
    if "full name" in name_lower or name_lower.strip() == "name":
        return f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"
    if "email" in name_lower:
        f = random.choice(_FIRST_NAMES).lower()
        l = random.choice(_LAST_NAMES).lower()
        return f"{f}.{l}{row_idx}@example.com"
    if "phone" in name_lower or "tel" in name_lower:
        return f"+1-555-{random.randint(1000,9999)}"
    if "city" in name_lower:
        return random.choice(_CITIES)
    if "country" in name_lower:
        return random.choice(["USA", "UK", "Germany", "France", "Japan", "Canada"])
    if "gender" in name_lower or "sex" in name_lower:
        return random.choice(_GENDERS)
    if "industry" in name_lower or "sector" in name_lower:
        return random.choice(_INDUSTRIES)
    if "category" in name_lower:
        return random.choice(_CATEGORIES)
    if "status" in name_lower:
        return random.choice(_STATUSES)
    if "type" in name_lower and "bool" not in name_lower:
        return random.choice(["A", "B", "C", "Standard", "Premium"])
    if "address" in name_lower:
        return f"{random.randint(100,999)} {random.choice(['Main','Oak','Elm','Park','Maple'])} St"
    if "color" in name_lower or "colour" in name_lower:
        return random.choice(["Red", "Blue", "Green", "Black", "White", "Yellow"])
    if "description" in name_lower or "note" in name_lower or "bio" in name_lower or "detail" in name_lower:
        words = ["Lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]
        return " ".join(random.sample(words, min(5, len(words))))

    # Type-based fallback
    if cat == "INT":
        # ID columns get sequential-ish values
        if "id" in name_lower:
            return row_idx + 1
        if "age" in name_lower:
            return random.randint(18, 80)
        if "quantity" in name_lower or "count" in name_lower or "num_" in name_lower:
            return random.randint(1, 100)
        if "year" in name_lower:
            return random.randint(2000, 2026)
        return random.randint(1, 1000)
    if cat == "DECIMAL":
        if "price" in name_lower or "amount" in name_lower or "total" in name_lower:
            return round(random.uniform(5, 5000), 2)
        if "salary" in name_lower or "fee" in name_lower:
            return round(random.uniform(30000, 150000), 2)
        if "percentage" in name_lower or "rate" in name_lower:
            return round(random.uniform(0.01, 99.99), 2)
        return round(random.uniform(1, 10000), 2)
    if cat == "DATE":
        y = random.randint(2018, 2026)
        m = random.randint(1, 12)
        d = random.randint(1, 28)
        return f"{y}-{m:02d}-{d:02d}"
    if cat == "BOOL":
        return random.choice([0, 1])
    if "url" in name_lower or "link" in name_lower:
        return f"https://example.com/{col_name}/{row_idx}"
    if "code" in name_lower or "ticker" in name_lower:
        return ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=random.randint(3,5)))
    # Default TEXT
    return f"{col_name}_{row_idx+1}"

def _type_to_sqlite(col_type: str) -> str:
    """Maps DDL types to SQLite-compatible types."""
    t = col_type.upper()
    if any(k in t for k in ["INT", "SERIAL", "BIGINT", "SMALLINT", "TINYINT"]):
        return "INTEGER"
    if any(k in t for k in ["DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL", "MONEY"]):
        return "REAL"
    if "BOOL" in t:
        return "INTEGER"
    return "TEXT"

def generate_sample_data(schema: dict, num_rows: int = 15) -> dict:
    """
    Generates sample data for all tables, respecting FK relationships.
    Returns: { table_name: [ {col: val}, ... ] }
    """
    data = {}
    fk_pools = {}

    # Determine table order (parents before children based on FK deps)
    table_order = list(schema.keys())
    # Simple topological sort: tables with no outgoing FKs come first
    depends_on = {}
    for tname, tinfo in schema.items():
        refs = set()
        for fk in tinfo["fks"]:
            refs.add(fk["ref_table"])
        for col in tinfo["columns"]:
            if col["is_fk"] and col["fk_ref"]:
                ref_table = col["fk_ref"].split(".")[0]
                refs.add(ref_table)
        depends_on[tname] = refs

    # Order: parents first
    ordered = []
    remaining = set(table_order)
    while remaining:
        ready = [t for t in remaining if not depends_on[t] or
                 all(r in [o for o in ordered] for r in depends_on[t] if r in schema)]
        if not ready:
            ordered.extend(remaining)
            break
        for t in ready:
            ordered.append(t)
            remaining.discard(t)

    for table_name in ordered:
        tinfo = schema[table_name]
        rows = []

        # Build FK pool for this table
        table_fk_pool = {}
        for col in tinfo["columns"]:
            if col["is_fk"] and col["fk_ref"]:
                ref_parts = col["fk_ref"].split(".")
                ref_table = ref_parts[0]
                ref_col = ref_parts[1] if len(ref_parts) > 1 else col["fk_ref"]
                if ref_table in data:
                    pool = [r.get(ref_col, r_idx+1) for r_idx, r in enumerate(data[ref_table])]
                    table_fk_pool[col["name"]] = pool

        for i in range(num_rows):
            row = {}
            for col in tinfo["columns"]:
                val = _generate_value(col["name"], col["type"], i, table_fk_pool)
                row[col["name"]] = val
            rows.append(row)

        data[table_name] = rows

    return data

# SQL Executor
def build_in_memory_db(schema: dict, sample_data: dict) -> sqlite3.Connection:
    """Builds an in-memory SQLite database from parsed schema and sample data."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")

    # Use sample_data order 
    table_order = list(sample_data.keys()) if sample_data else list(schema.keys())
    for table_name in table_order:
        tinfo = schema.get(table_name)
        if tinfo is None:
            continue
        cols_ddl = []
        for col in tinfo["columns"]:
            sqlite_type = _type_to_sqlite(col["type"])
            clause = f"{col['name']} {sqlite_type}"
            if col["is_pk"]:
                clause += " PRIMARY KEY"
            cols_ddl.append(clause)

        # Add FK constraints
        for fk in tinfo["fks"]:
            cols_ddl.append(
                f"FOREIGN KEY ({fk['column']}) REFERENCES {fk['ref_table']}({fk['ref_column']})"
            )

        ddl = f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(cols_ddl) + "\n)"
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError as e:
            st.warning(f"Could not create table `{table_name}`: {e}")
            continue

        # Insert sample data
        if table_name in sample_data:
            inserted, failed = 0, 0
            for row in sample_data[table_name]:
                columns = list(row.keys())
                placeholders = ", ".join(["?" for _ in columns])
                values = [row[c] for c in columns]
                try:
                    conn.execute(
                        f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})",
                        values
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    failed += 1  # FK/PK violation — skip this row
                except Exception:
                    failed += 1
            if failed:
                st.caption(
                    f" `{table_name}`: {failed} of {inserted + failed} sample row(s) skipped "
                    f"(constraint violations) — {inserted} row(s) loaded."
                )

    conn.commit()
    return conn

def execute_sql(conn: sqlite3.Connection, sql: str) -> tuple:
    """Executes SQL and returns (columns, rows) or (None, error_msg)."""
    if not sql or not sql.strip():
        return None, "Empty SQL query."
    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return columns, rows
    except Exception as e:
        return None, str(e)

def read_schema_from_sqlite(db_path: str) -> str:
    """Reads DDL schema from an existing SQLite database file."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        # sqlite_master.sql holds the CREATE TABLE text exactly as it was
        # originally executed — SQLite does NOT store a trailing semicolon.
        # parse_ddl() (and the LLM prompt) expect ';'-terminated statements,
        # so normalize each one here. Each row is already a single complete
        # statement with balanced parens, so this is safe to do per-row.
        statements = [row[0].strip() for row in cursor.fetchall() if row[0]]
        statements = [s if s.endswith(';') else s + ';' for s in statements]
        conn.close()
        return "\n\n".join(statements)
    except Exception as e:
        return f"-- Error reading schema: {e}"

# Schema Visualization Helpers
def build_er_diagram(schema: dict) -> str:
    """Builds a Mermaid erDiagram from structured schema."""
    if not schema:
        return ""

    lines = ["erDiagram"]

    for tname, tinfo in schema.items():
        # Uppercase entity name for Mermaid compatibility
        entity = tname.upper()

        col_defs = []
        for col in tinfo["columns"]:
            col_type = col["type"].upper()
            # Map to Mermaid-compatible types (capitalized)
            if any(k in col_type for k in ["INT", "SERIAL", "BIGINT", "SMALLINT"]):
                m_type = "Int"
            elif any(k in col_type for k in ["DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL"]):
                m_type = "Float"
            elif any(k in col_type for k in ["DATE", "TIME", "TIMESTAMP", "DATETIME"]):
                m_type = "DateTime"
            elif "BOOL" in col_type:
                m_type = "Bool"
            else:
                m_type = "String"

            pk_fk_attrs = []
            if col.get("is_pk"):
                pk_fk_attrs.append("PK")
            if col.get("is_fk"):
                pk_fk_attrs.append("FK")
            
            attrs_str = " " + ",".join(pk_fk_attrs) if pk_fk_attrs else ""
            col_defs.append(f"    {m_type} {col['name']}{attrs_str}")

        lines.append(f"  {entity} {{")
        lines.extend(col_defs)
        lines.append("  }")

        # Collect relationships
        for fk in tinfo["fks"]:
            ref_entity = fk["ref_table"].upper()
            # Parent (ref_entity) has many children (entity)
            lines.append(f"  {ref_entity} ||--o{{ {entity} : \"{fk['column']}\"")

    return "\n".join(lines)

def extract_tables_from_sql(sql: str, schema: dict) -> list:
    """
    Finds which known schema tables are referenced in a SQL query's FROM/JOIN
    clauses (case-insensitive). Falls back to every table in the schema if
    nothing matches (e.g. parsing failed) so callers always get something
    reasonable to diagram.
    """
    if not schema:
        return []
    if not sql:
        return list(schema.keys())

    name_lookup = {name.lower(): name for name in schema.keys()}
    referenced = []
    for m in re.finditer(r'\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)', sql, re.IGNORECASE):
        candidate = m.group(1).lower()
        if candidate in name_lookup and name_lookup[candidate] not in referenced:
            referenced.append(name_lookup[candidate])

    return referenced if referenced else list(schema.keys())

def build_er_diagram_scoped(schema: dict, table_names: list) -> str:
    """
    Builds a Mermaid erDiagram restricted to the given tables. FK edges that
    would point to a table outside the given set are dropped, so the diagram
    never references an entity it doesn't also define.
    """
    if not schema or not table_names:
        return ""

    name_set = {t.lower() for t in table_names}
    scoped = OrderedDict()
    for tname, tinfo in schema.items():
        if tname.lower() not in name_set:
            continue
        fks_in_scope = [fk for fk in tinfo["fks"] if fk["ref_table"].lower() in name_set]
        scoped[tname] = {"columns": tinfo["columns"], "fks": fks_in_scope}

    return build_er_diagram(scoped)


def _mermaid_node_text(text: str, max_len: int = 70) -> str:
    """Sanitizes and truncates text for safe use inside a quoted Mermaid node label."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('"', "'")
    # Backtick is a special character in Mermaid 10.x — a pair of backticks
    # switches the label into "markdown string" mode (bold/italic/<br>
    # parsing). SQL using MySQL-style backtick-quoted identifiers could
    # otherwise produce unexpected formatting inside the rendered diagram.
    text = text.replace('`', "'")
    if len(text) > max_len:
        text = text[:max_len - 1].rstrip() + "…"
    return text

def _split_sql_clauses(sql: str) -> "OrderedDict[str, str]":
    """
    Lightweight, paren-depth-aware splitter that pulls out the main top-level
    clauses of a single SELECT statement (SELECT/FROM/WHERE/GROUP BY/HAVING/
    ORDER BY/LIMIT). Not a full SQL parser — subquery internals are treated as
    opaque text and not split into their own clauses, which is the right
    behavior here since we only want the outer query's execution shape.
    """
    keywords = ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"]
    upper_sql = sql.upper()
    n = len(sql)
    depth = 0
    positions = []
    i = 0
    while i < n:
        ch = sql[i]
        if ch == '(':
            depth += 1
            i += 1
            continue
        if ch == ')':
            depth -= 1
            i += 1
            continue
        if depth == 0:
            for kw in keywords:
                kw_len = len(kw)
                if upper_sql[i:i + kw_len] == kw:
                    before_ok = (i == 0) or not (sql[i - 1].isalnum() or sql[i - 1] == '_')
                    after_idx = i + kw_len
                    after_ok = (after_idx >= n) or not (sql[after_idx].isalnum() or sql[after_idx] == '_')
                    if before_ok and after_ok:
                        positions.append((kw, i))
                        i += kw_len
                        break
            else:
                i += 1
                continue
            continue
        i += 1

    clauses = OrderedDict()
    for idx, (kw, start) in enumerate(positions):
        end = positions[idx + 1][1] if idx + 1 < len(positions) else n
        clause_text = sql[start + len(kw):end].strip().rstrip(';').strip()
        if kw not in clauses:
            clauses[kw] = clause_text
    return clauses


def build_execution_flowchart(sql: str) -> str:
    """
    Builds a Mermaid flowchart (TD) representing the logical execution order
    of a SQL query, derived directly from the SQL text via _split_sql_clauses
    — no model call involved, so this always works as long as the SQL parses.
    """
    if not sql or not sql.strip():
        return ""

    clauses = _split_sql_clauses(sql)
    if "FROM" not in clauses and "SELECT" not in clauses:
        return ""

    nodes = []   # (id, label)
    edges = []   # (from_id, to_id)
    node_counter = [0]

    def new_node(label: str) -> str:
        node_counter[0] += 1
        nid = f"N{node_counter[0]}"
        nodes.append((nid, label))
        return nid

    prev_id = None

    def chain(nid: str):
        nonlocal prev_id
        if prev_id is not None:
            edges.append((prev_id, nid))
        prev_id = nid

    # FROM + JOINs
    from_text = clauses.get("FROM", "")
    if from_text:
        join_pattern = re.compile(
            r'\b(INNER\s+JOIN|LEFT\s+(?:OUTER\s+)?JOIN|RIGHT\s+(?:OUTER\s+)?JOIN|'
            r'FULL\s+(?:OUTER\s+)?JOIN|CROSS\s+JOIN|JOIN)\b',
            re.IGNORECASE
        )
        parts = join_pattern.split(from_text)
        base_table = parts[0].strip()
        nid = new_node(f'FROM {_mermaid_node_text(base_table)}')
        chain(nid)
        # parts alternates: [base, join_kw, join_clause, join_kw, join_clause, ...]
        for j in range(1, len(parts) - 1, 2):
            join_kw = re.sub(r'\s+', ' ', parts[j].strip().upper())
            join_clause = parts[j + 1].strip()
            nid = new_node(f'{join_kw} {_mermaid_node_text(join_clause)}')
            chain(nid)

    # WHERE
    if clauses.get("WHERE"):
        nid = new_node(f'WHERE {_mermaid_node_text(clauses["WHERE"])}')
        chain(nid)

    # GROUP BY
    if clauses.get("GROUP BY"):
        nid = new_node(f'GROUP BY {_mermaid_node_text(clauses["GROUP BY"])}')
        chain(nid)

    # HAVING
    if clauses.get("HAVING"):
        nid = new_node(f'HAVING {_mermaid_node_text(clauses["HAVING"])}')
        chain(nid)

    # SELECT (projection) — shown after filtering, matching SQL's logical
    # (not textual) execution order
    select_text = clauses.get("SELECT", "")
    if select_text:
        label = "SELECT DISTINCT" if select_text.upper().startswith("DISTINCT") else "SELECT"
        cols = re.sub(r'^DISTINCT\s+', '', select_text, flags=re.IGNORECASE)
        nid = new_node(f'{label} {_mermaid_node_text(cols)}')
        chain(nid)

    # ORDER BY
    if clauses.get("ORDER BY"):
        nid = new_node(f'ORDER BY {_mermaid_node_text(clauses["ORDER BY"])}')
        chain(nid)

    # LIMIT
    if clauses.get("LIMIT"):
        nid = new_node(f'LIMIT {_mermaid_node_text(clauses["LIMIT"])}')
        chain(nid)

    if not nodes:
        return ""

    # Final "results" node
    result_id = new_node('Return Results')
    chain(result_id)

    lines = ["flowchart TD"]
    for nid, label in nodes:
        lines.append(f'  {nid}["{label}"]')
    for a, b in edges:
        lines.append(f'  {a} --> {b}')

    return "\n".join(lines)


def render_schema_cards(schema: dict):
    """Renders schema tables as visual cards."""
    if not schema:
        st.info("No tables detected. Enter DDL in the Query tab first.")
        return

    # Stats row
    total_cols = sum(len(t["columns"]) for t in schema.values())
    total_fks = sum(len(t["fks"]) for t in schema.values())

    cols_stat = st.columns(4)
    cols_stat[0].metric("Tables", len(schema))
    cols_stat[1].metric("Columns", total_cols)
    cols_stat[2].metric("Foreign Keys", total_fks)

    col_type_map = {}
    for tinfo in schema.values():
        for col in tinfo["columns"]:
            typ = col["type"].upper()
            key = "INT" if any(k in typ for k in ["INT","SERIAL"]) else \
                  "DECIMAL" if any(k in typ for k in ["DECIMAL","FLOAT","DOUBLE","REAL"]) else \
                  "DATE" if "DATE" in typ or "TIME" in typ else \
                  "BOOL" if "BOOL" in typ else "TEXT"
            col_type_map[key] = col_type_map.get(key, 0) + 1
    cols_stat[3].metric("Types", len(col_type_map))

    st.divider()

    # Table cards
    cols = st.columns(min(len(schema), 3))
    for idx, (tname, tinfo) in enumerate(schema.items()):
        with cols[idx % len(cols)]:
            # Build complete card HTML in one string
            col_html_parts = []
            for col in tinfo["columns"]:
                badges = []
                if col["is_pk"]:
                    badges.append("<span class='qm-badge qm-badge-pk'>PK</span>")
                if col["is_fk"]:
                    badges.append("<span class='qm-badge qm-badge-fk'>FK</span>")

                type_badge_class = {
                    "INT": "qm-badge-int", "SERIAL": "qm-badge-int",
                    "BIGINT": "qm-badge-int", "SMALLINT": "qm-badge-int",
                    "DECIMAL": "qm-badge-dec", "FLOAT": "qm-badge-dec",
                    "DOUBLE": "qm-badge-dec", "REAL": "qm-badge-dec",
                    "DATE": "qm-badge-date", "TIME": "qm-badge-date",
                    "TIMESTAMP": "qm-badge-date", "DATETIME": "qm-badge-date",
                    "BOOL": "qm-badge-bool",
                }.get(col["type"].upper().split("(")[0], "qm-badge-txt")

                badges.append(
                    f"<span class='qm-badge {type_badge_class}'>{col['type']}</span>"
                )

                fk_hint = ""
                if col["is_fk"] and col["fk_ref"]:
                    fk_hint = f" <span class='qm-fk-arrow'>→ {col['fk_ref']}</span>"

                col_html_parts.append(
                    f"<div class='qm-col-row'>"
                    f"  <span>{col['name']}</span>"
                    f"  <span class='qm-col-type'>{col['type']}</span>"
                    f"  {' '.join(badges)}{fk_hint}"
                    f"</div>"
                )

            card_html = (
                f"<div class='qm-table-card'>"
                f"<div class='qm-table-name'> {tname}</div>"
                f"{''.join(col_html_parts)}"
                f"</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)


def render_mermaid(code: str, height: int = 380):
    """Renders Mermaid.js diagram safely using Base64 injection to prevent HTML escaping issues."""
    if not code or not code.strip():
        return

    import base64
    b64_code = base64.b64encode(code.encode('utf-8')).decode('utf-8')
    
    html_block = f"""
    <div id="mermaid-output" style="background:#ffffff;border-radius:12px;padding:1rem;overflow:auto;height:100%;font-family:sans-serif;">
        <span style="color:#666;">Rendering diagram...</span>
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: false, theme: "default", securityLevel: "loose" }});
      
      async function drawDiagram() {{
          const output = document.getElementById('mermaid-output');
          let rawCode = "";
          try {{
              rawCode = decodeURIComponent(escape(window.atob("{b64_code}")));
              const {{ svg }} = await mermaid.render('mermaid-svg', rawCode);
              output.innerHTML = svg;
          }} catch (err) {{
              output.innerHTML = 
                  `<div style="color:#d93025; font-weight:bold; margin-bottom:10px;">⚠️ Mermaid Syntax Error (AI Generated)</div>` + 
                  `<pre style="background:#fce8e6; padding:10px; border-radius:5px; font-size:12px; overflow:auto; color:#d93025;">${{err.message}}</pre>` +
                  `<div style="margin-top:15px; font-weight:bold;">Problematic Code Generated by Model:</div>` +
                  `<pre style="background:#f1f3f4; padding:10px; border-radius:5px; font-size:12px; overflow:auto; color:#333;">${{rawCode}}</pre>`;
          }}
      }}
      drawDiagram();
    </script>
    """
    st.components.v1.html(html_block, height=height, scrolling=True)


def compare_results(gen_cols, gen_rows, gold_cols, gold_rows, order_sensitive: bool = False) -> dict:
    """
    Compares the generated query's result set against the gold query's result set.
    By default this is order-insensitive but duplicate-aware (multiset comparison),
    matching standard SQL execution-accuracy evaluation. If order_sensitive=True
    (used when the gold SQL contains an ORDER BY), row order must match exactly.
    """
    gen_rows = gen_rows or []
    gold_rows = gold_rows or []
    gen_tuples = [tuple(r) for r in gen_rows]
    gold_tuples = [tuple(r) for r in gold_rows]

    if order_sensitive:
        match = gen_tuples == gold_tuples
    else:
        match = Counter(gen_tuples) == Counter(gold_tuples)

    only_in_generated, only_in_gold = [], []
    if not match:
        gen_multiset = Counter(gen_tuples)
        gold_multiset = Counter(gold_tuples)
        only_in_generated = list((gen_multiset - gold_multiset).elements())
        only_in_gold = list((gold_multiset - gen_multiset).elements())

    return {
        "match": match,
        "only_in_generated": only_in_generated,
        "only_in_gold": only_in_gold,
        "columns_match": list(gen_cols or []) == list(gold_cols or []),
    }


def build_full_report(question, schema_text, sql, gen_time, query_cols, query_results, exec_error,
                       er_code, flow_code, gold_sql, gold_cols, gold_rows, gold_error) -> str:
    """Builds a single Markdown report combining the question, SQL, diagrams, results, and accuracy check."""
    lines = ["# QueryMaster Report", "", f"**Question:** {question}", ""]

    lines += ["## Schema", "```sql", schema_text.strip(), "```", ""]

    lines += ["## Generated SQL", f"_Generation time: {gen_time:.2f}s_", "```sql", sql.strip(), "```", ""]

    if exec_error:
        lines.append(f"**Execution error:** {exec_error}")
    elif query_cols is not None:
        n_rows = len(query_results) if isinstance(query_results, list) else 0
        lines.append(f"## Query Results ({n_rows} row(s))")
        if query_results:
            lines.append("| " + " | ".join(query_cols) + " |")
            lines.append("|" + "|".join(["---"] * len(query_cols)) + "|")
            for row in query_results[:25]:
                lines.append("| " + " | ".join(str(v) for v in row) + " |")
            if n_rows > 25:
                lines.append(f"_... and {n_rows - 25} more row(s)_")
        else:
            lines.append("_No rows returned._")
    lines.append("")

    if er_code:
        lines += ["## ER Diagram (Mermaid)", "```mermaid", er_code, "```", ""]
    if flow_code:
        lines += ["## Execution Flow (Mermaid)", "```mermaid", flow_code, "```", ""]

    if gold_sql:
        lines += ["## Execution Accuracy Check", "**Gold SQL:**", "```sql", gold_sql.strip(), "```"]
        if gold_error:
            lines.append(f"**Gold SQL failed to execute:** {gold_error}")
        elif exec_error:
            lines.append("_Couldn't compare — generated SQL failed to execute._")
        else:
            order_sensitive = "order by" in gold_sql.lower()
            verdict = compare_results(query_cols, query_results, gold_cols, gold_rows, order_sensitive=order_sensitive)
            if verdict["match"]:
                n = len(query_results) if isinstance(query_results, list) else 0
                lines.append(f"Result: Execution match — {n} row(s).")
            else:
                lines.append(
                    f"Result: Execution mismatch — "
                    f"{len(verdict['only_in_generated'])} row(s) only in generated output, "
                    f"{len(verdict['only_in_gold'])} row(s) only in gold output."
                )

    return "\n".join(lines)


def render_results_tab(results_cols, results_rows, exec_error=None):
    """Renders query execution results with export options."""
    if exec_error:
        st.error(f"SQL execution error: {exec_error}")
        return

    if results_cols is None:
        return

    if not results_rows:
        st.info("Query executed successfully but returned no rows (empty result set).")
        return

    st.subheader(f"📋 Query Results ({len(results_rows)} row{'s' if len(results_rows) != 1 else ''})")

    df = pd.DataFrame(results_rows, columns=results_cols)

    # Scrollable table
    st.dataframe(df, use_container_width=True, height=min(400, 35 * len(df) + 38))

    # Export buttons
    col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 4])
    with col_exp1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "CSV", csv_data, "query_results.csv", "text/csv",
            use_container_width=True
        )
    with col_exp2:
        json_data = df.to_json(orient="records", indent=2).encode("utf-8")
        st.download_button(
            "JSON", json_data, "query_results.json", "application/json",
            use_container_width=True
        )


# Model Loading
@st.cache_resource
def load_model():
    model_id = "mlx-community/Qwen3-8B-4bit"
    # Resolve relative to this script's location, not the shell's current
    # working directory — otherwise launching with `streamlit run` from a
    # different folder silently breaks the adapter lookup.
    adapter_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapters_best")
    if not os.path.isdir(adapter_path):
        st.error(
            f"Adapter folder not found at `{adapter_path}`. "
            "Make sure `adapters_best/` sits next to app.py, or update the path in `load_model()`."
        )
        st.stop()
    return load(model_id, adapter_path=adapter_path)



# Session State Init
def init_session():
    defaults = {
        "schema_text": "",
        "parsed_schema": {},
        "generated_sql": "",
        "query_results": None,
        "query_cols": None,
        "exec_error": None,
        "er_code": "",
        "flow_code": "",
        "diagram_truncated": False,
        "last_raw_response": "",
        "gen_time": 0.0,
        "query_history": [],
        "sample_rows": 15,
        "use_uploaded_db": False,
        "uploaded_db_path": None,
        "uploaded_db_name": None,
        "gold_cols": None,
        "gold_rows": None,
        "gold_error": None,
        "gold_sql_used": "",
        "generation_error": None,
        "uploader_key": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# Main App 
def main():
    init_session()

    # SIDEBAR
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center; margin-bottom:0.5rem;'>"
            "<span style='font-size:1.5rem;'></span>"
            "<h3 style='margin:0;'>QueryMaster</h3>"
            "<p style='font-size:0.75rem; opacity:0.6;'>Text-to-SQL Agent</p>"
            "</div>",
            unsafe_allow_html=True
        )
        st.divider()

        # Database Upload 
        st.subheader("Data Source")
        uploaded_file = st.file_uploader(
            "Upload SQLite database (optional)",
            type=["db", "sqlite", "sqlite3"],
            help="Upload a .db file to use real data instead of generated samples.",
            key=f"db_uploader_{st.session_state.uploader_key}",
        )
        if uploaded_file:
            tmp_path = f"/tmp/qm_uploaded_{hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]}.db"
            # Only touch disk / overwrite the schema box on a genuinely new upload —
            # otherwise every rerun (e.g. moving the sample-rows slider) would re-write
            # the temp file and stomp any manual edits the user made to the schema.
            if tmp_path != st.session_state.uploaded_db_path:
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.uploaded_db_path = tmp_path
                st.session_state.uploaded_db_name = uploaded_file.name
                st.session_state.use_uploaded_db = True

                schema_from_db = read_schema_from_sqlite(tmp_path)
                st.session_state.schema_text = schema_from_db
                # Directly set the widget's own key so the textarea actually
                # refreshes — Streamlit ignores `value=` once a keyed widget
                # already has a stored value.
                st.session_state["schema_textarea"] = schema_from_db

        if st.session_state.uploaded_db_name:
            st.success(f"{st.session_state.uploaded_db_name}")
            if st.button("Clear uploaded DB", use_container_width=True):
                st.session_state.uploaded_db_path = None
                st.session_state.uploaded_db_name = None
                st.session_state.use_uploaded_db = False
                st.session_state.schema_text = ""
                st.session_state["schema_textarea"] = ""
                st.session_state.uploader_key += 1  # forces a fresh, empty uploader widget
                st.rerun()

        st.divider()

        # Sample Data Rows
        st.subheader("Settings")
        st.session_state.sample_rows = st.slider(
            "Sample rows per table",
            min_value=5, max_value=50, value=st.session_state.sample_rows, step=5,
            help="Number of synthetic rows to generate for each table (ignored if using uploaded DB)."
        )

        st.divider()

        # Query History
        st.subheader("Query History")
        if st.session_state.query_history:
            for i, entry in enumerate(reversed(st.session_state.query_history[-10:])):
                with st.expander(f"{entry['question'][:50]}{'...' if len(entry['question'])>50 else ''}", expanded=False):
                    st.caption(f"**SQL:** `{entry['sql'][:100]}`")
                    st.caption(f"Time: {entry['time']:.2f}s | Rows: {entry.get('rows', 'N/A')}")
                    if st.button("Reload", key=f"hist_{i}", use_container_width=True):
                        # Set the widgets' own keys directly — Streamlit ignores a
                        # text_area's `value=` once the key already holds a value,
                        # so writing only to the tracking vars wouldn't actually
                        # update what's shown on screen.
                        st.session_state["schema_textarea"] = entry.get("schema", "")
                        st.session_state["question_input"] = entry.get("question", "")
                        st.session_state["gold_sql_textarea"] = entry.get("gold_sql", "")

                        st.session_state.schema_text = entry.get("schema", "")
                        st.session_state.generated_sql = entry["sql"]
                        st.session_state.gen_time = entry.get("time", 0.0)
                        st.session_state.query_cols = entry.get("query_cols")
                        st.session_state.query_results = entry.get("query_results")
                        st.session_state.exec_error = entry.get("exec_error")
                        st.session_state.er_code = entry.get("er_code")
                        st.session_state.flow_code = entry.get("flow_code")
                        st.session_state.diagram_truncated = entry.get("diagram_truncated", False)
                        st.session_state.gold_cols = entry.get("gold_cols")
                        st.session_state.gold_rows = entry.get("gold_rows")
                        st.session_state.gold_error = entry.get("gold_error")
                        st.session_state.gold_sql_used = entry.get("gold_sql", "")
                        st.session_state.generation_error = None
                        st.session_state.last_raw_response = ""  # not persisted per history entry
                        st.rerun()
            if st.button("Clear history", use_container_width=True):
                st.session_state.query_history = []
                st.rerun()
        else:
            st.caption("No queries yet. Generate one!")

        st.divider()

        # Info 
        with st.expander("About"):
            st.markdown("""
**QueryMaster v2.0** — Text-to-SQL AI Agent

- **Model:** Qwen3-8B-4bit + LoRA
- **Benchmark:** Spider 1.0 dev — 67.4% EX
- **Datasets:** BIRD, SynSQL, Gretel

Built for Final Project
            """)

    # CSS injected AFTER sidebar
    inject_css()

    # MAIN CONTENT
    st.title("QueryMaster: End-to-End Text-to-SQL Agent")
    st.markdown(
        "<p style='opacity:0.6; margin-top:-10px;'>"
        "FEE306 Applied Artificial Neural Networks — Final Project"
        "</p>",
        unsafe_allow_html=True
    )

    # Model Loading
    with st.spinner("Loading model... (Apple Silicon Unified Memory)"):
        model, tokenizer = load_model()
    st.success("Qwen3-8B-4bit + LoRA adapters loaded")

    # TABS
    tab1, tab2 = st.tabs(["Query", "Schema Visualizer"])

    # TAB 1: QUERY
    with tab1:
        col_input1, col_input2 = st.columns([1, 1])

        with col_input1:
            with st.container(border=True):
                st.subheader("Database Schema (DDL)")

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

                schema_text = st.text_area(
                    "SQL schema:",
                    value=st.session_state.schema_text or default_schema,
                    height=350,
                    key="schema_textarea",
                )
                st.session_state.schema_text = schema_text

                # Parse schema in real-time
                parsed = parse_ddl(schema_text)
                st.session_state.parsed_schema = parsed

                if parsed:
                    total_tables = len(parsed)
                    total_cols = sum(len(t["columns"]) for t in parsed.values())
                    st.markdown(
                        f"<div class='qm-stats'>"
                        f"<span class='qm-stat'><span class='qm-stat-value'>{total_tables}</span> tables</span>"
                        f"<span class='qm-stat'><span class='qm-stat-value'>{total_cols}</span> columns</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.caption("Enter CREATE TABLE DDL above — parsed schema will appear here.")

        with col_input2:
            with st.container(border=True):
                st.subheader("User Question")
                question = st.text_area(
                    "What do you want to learn from the database?",
                    value="Find the first name and city of customers who have an order with a total amount greater than 4000, ordered by total amount descending.",
                    height=100,
                    key="question_input",
                )

                st.markdown("<br>", unsafe_allow_html=True)
                gold_sql_input = st.text_area(
                    "Expected Gold SQL (optional):",
                    value="",
                    height=80,
                    key="gold_sql_textarea",
                    help="Paste the correct SQL here to compare with the model's output.",
                )

                st.markdown("<br>", unsafe_allow_html=True)
                gen_btn_col1, gen_btn_col2 = st.columns([2, 1])
                with gen_btn_col1:
                    generate_btn = st.button(
                        "Generate SQL", use_container_width=True, type="primary"
                    )
                with gen_btn_col2:
                    regen_btn = st.button(
                        "Regenerate", use_container_width=True,
                        disabled=not st.session_state.generated_sql,
                        help="Re-run with the same question & schema — useful if the last output wasn't great."
                    )

        # SQL Generation
        if generate_btn or regen_btn:
            if not schema_text.strip():
                st.warning("Please enter a database schema first.")
            elif not question.strip():
                st.warning("Please enter a question.")
            else:
                st.session_state.generation_error = None
                response = None
                with st.spinner("Model is thinking, generating SQL..."):
                    messages = [
                        {"role": "system", "content": """You are a senior SQLite data analyst. Given the database schema below, generate an accurate SQL query that answers the user's question. 
CRITICAL INSTRUCTIONS:
1. Use SQLite syntax.
2. Pay close attention to FOREIGN KEY relationships to ensure you JOIN the correct tables. Do not skip intermediate tables (e.g. join artists to albums, then albums to tracks).
3. Be extremely careful about WHICH column you apply filters to (e.g. do not filter on Album Title when the user asks for an Artist Name).
Wrap the query in ```sql ... ``` blocks."""},
                        {"role": "user", "content": f"Database Schema:\n{schema_text}\n\nQuestion: {question}"}
                    ]

                    try:
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        start_time = time.perf_counter()
                        response = generate(model, tokenizer, prompt=prompt, max_tokens=4096, verbose=False)
                        end_time = time.perf_counter()
                        gen_time = end_time - start_time
                    except Exception as e:
                        st.session_state.generation_error = str(e)
                        response = None

                if response is not None:
                    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                    st.session_state.last_raw_response = response

                    # Parse SQL
                    sql_match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
                    if sql_match:
                        sql = sql_match.group(1).strip()
                    else:
                        select_match = re.search(r"(SELECT\s+.*?)(\n```|$|<\|im_end\|>)", response, re.DOTALL | re.IGNORECASE)
                        sql = select_match.group(1).strip() if select_match else response.strip()
                    sql = sql.rstrip(";").strip()

                    # ER diagram & execution flowchart are now built deterministically
                    # from the parsed schema + generated SQL, rather than relying on
                    # the fine-tuned (SQL-only) model to also emit Mermaid blocks — the
                    # LoRA adapter was trained exclusively on question→SQL pairs and
                    # reliably stops generating right after the SQL, so asking it for
                    # extra output via the system prompt doesn't work in practice.
                    tables_used = extract_tables_from_sql(sql, st.session_state.parsed_schema)
                    er_code = build_er_diagram_scoped(st.session_state.parsed_schema, tables_used)
                    flow_code = build_execution_flowchart(sql)

                    st.session_state.generated_sql = sql
                    st.session_state.er_code = er_code
                    st.session_state.flow_code = flow_code
                    st.session_state.gen_time = gen_time
                    st.session_state.diagram_truncated = False

                    # ── Execute SQL (generated query + optional gold query for comparison) ──
                    gold_sql_clean = gold_sql_input.strip()
                    st.session_state.gold_cols = None
                    st.session_state.gold_rows = None
                    st.session_state.gold_error = None
                    st.session_state.gold_sql_used = gold_sql_clean

                    if st.session_state.use_uploaded_db and st.session_state.uploaded_db_path:
                        try:
                            conn = sqlite3.connect(st.session_state.uploaded_db_path)
                            cols, rows = execute_sql(conn, sql)
                            st.session_state.query_cols = cols
                            st.session_state.query_results = rows
                            st.session_state.exec_error = rows if isinstance(rows, str) else None
                            if gold_sql_clean:
                                g_cols, g_rows = execute_sql(conn, gold_sql_clean)
                                st.session_state.gold_cols = g_cols
                                st.session_state.gold_rows = g_rows
                                st.session_state.gold_error = g_rows if isinstance(g_rows, str) else None
                            conn.close()
                        except Exception as e:
                            st.session_state.query_cols = None
                            st.session_state.query_results = None
                            st.session_state.exec_error = str(e)
                    else:
                        parsed = st.session_state.parsed_schema
                        if parsed:
                            sample_data = generate_sample_data(parsed, st.session_state.sample_rows)
                            conn = build_in_memory_db(parsed, sample_data)
                            cols, rows = execute_sql(conn, sql)
                            st.session_state.query_cols = cols
                            st.session_state.query_results = rows
                            st.session_state.exec_error = rows if isinstance(rows, str) else None
                            if gold_sql_clean:
                                g_cols, g_rows = execute_sql(conn, gold_sql_clean)
                                st.session_state.gold_cols = g_cols
                                st.session_state.gold_rows = g_rows
                                st.session_state.gold_error = g_rows if isinstance(g_rows, str) else None
                            conn.close()
                        else:
                            st.session_state.query_cols = None
                            st.session_state.query_results = None
                            st.session_state.exec_error = "Could not parse schema for execution."

                    # Add to history (full snapshot so Reload can restore everything)
                    hist_entry = {
                        "question": question,
                        "sql": sql,
                        "schema": schema_text,
                        "time": gen_time,
                        "rows": len(st.session_state.query_results) if isinstance(st.session_state.query_results, list) else 0,
                        "gold_sql": gold_sql_clean,
                        "query_cols": st.session_state.query_cols,
                        "query_results": st.session_state.query_results,
                        "exec_error": st.session_state.exec_error,
                        "er_code": st.session_state.er_code,
                        "flow_code": st.session_state.flow_code,
                        "diagram_truncated": st.session_state.diagram_truncated,
                        "gold_cols": st.session_state.gold_cols,
                        "gold_rows": st.session_state.gold_rows,
                        "gold_error": st.session_state.gold_error,
                    }
                    st.session_state.query_history.append(hist_entry)

            st.rerun()

        if st.session_state.generation_error:
            st.error(
                f"Generation failed: {st.session_state.generation_error}\n\n"
                "This usually means the model/adapter didn't load correctly or ran out of memory. "
                "Try again, or restart the app if it persists."
            )

        #Display Results
        if st.session_state.generated_sql:
            st.divider()

            # SQL output with stats
            sql_col, time_col = st.columns([3, 1])
            with sql_col:
                st.subheader("Generated SQL")
                st.code(st.session_state.generated_sql, language="sql")
            with time_col:
                st.metric("Generation Time", f"{st.session_state.gen_time:.2f}s")

            # SQL download & copy
            dl_col1, dl_col2, dl_col3, _ = st.columns([1, 1, 1, 1])
            with dl_col1:
                st.download_button(
                    "Download .sql",
                    st.session_state.generated_sql,
                    "query.sql",
                    "text/plain",
                    use_container_width=True
                )
            with dl_col2:
                # Copy button via HTML
                escaped_sql = st.session_state.generated_sql.replace("`", "\\`").replace("$", "\\$")
                copy_html = f"""
                <button onclick="navigator.clipboard.writeText(`{escaped_sql}`)"
                        style="width:100%; padding:0.4rem 1rem; border-radius:8px;
                               border:1px solid rgba(255,255,255,0.15);
                               background:rgba(255,255,255,0.05); color:inherit;
                               cursor:pointer; font-size:0.85rem;">
                    Copy SQL
                </button>
                """
                st.components.v1.html(copy_html, height=40)
            with dl_col3:
                report_md = build_full_report(
                    question, schema_text, st.session_state.generated_sql, st.session_state.gen_time,
                    st.session_state.query_cols, st.session_state.query_results, st.session_state.exec_error,
                    st.session_state.er_code, st.session_state.flow_code,
                    st.session_state.gold_sql_used, st.session_state.gold_cols, st.session_state.gold_rows,
                    st.session_state.gold_error,
                )
                st.download_button(
                    "Full Report",
                    report_md,
                    "querymaster_report.md",
                    "text/markdown",
                    use_container_width=True,
                    help="Question, schema, SQL, diagrams, results, and accuracy check in one Markdown file."
                )

            # Query Results
            if st.session_state.query_cols is not None or st.session_state.exec_error:
                st.divider()
                render_results_tab(
                    st.session_state.query_cols,
                    st.session_state.query_results,
                    st.session_state.exec_error
                )

            # ER Diagram & Execution Flow — built deterministically from the
            # parsed schema + generated SQL (see build_er_diagram_scoped /
            # build_execution_flowchart), not from the model.
            if st.session_state.generated_sql:
                st.divider()
                st.subheader("Visual Breakdown")
                if not st.session_state.er_code and not st.session_state.flow_code:
                    st.info(
                        "Couldn't build a diagram for this query — usually means the "
                        "schema text above didn't parse into any tables. Check the "
                        "Schema Visualizer tab to confirm your DDL is being recognized."
                    )
                diag_tab1, diag_tab2 = st.tabs(["ER Diagram", "Execution Flow"])
                with diag_tab1:
                    if st.session_state.er_code:
                        render_mermaid(st.session_state.er_code, height=380)
                    else:
                        st.caption("No ER diagram available for this query.")
                with diag_tab2:
                    if st.session_state.flow_code:
                        render_mermaid(st.session_state.flow_code, height=380)
                    else:
                        st.caption("No execution flow chart available for this query.")

            # Debug: raw model output, kept for diagnosing SQL parsing issues.
            if st.session_state.last_raw_response:
                with st.expander("🔍 Raw model output (debug)", expanded=False):
                    st.text_area(
                        "Full unparsed response from the model:",
                        value=st.session_state.last_raw_response,
                        height=300,
                        key="debug_raw_response_view",
                    )

            # Gold SQL comparison — actually executes the gold query and checks for an execution match
            if st.session_state.gold_sql_used:
                st.divider()
                st.subheader("Execution Accuracy Check")

                if gold_sql_input.strip() != st.session_state.gold_sql_used:
                    st.warning(
                        "The gold SQL has changed since the last generation — "
                        "click Generate SQL again to refresh this comparison."
                    )

                st.code(st.session_state.gold_sql_used, language="sql")

                if st.session_state.gold_error:
                    st.error(f"Gold SQL failed to execute: {st.session_state.gold_error}")
                elif st.session_state.exec_error:
                    st.warning("Can't compare — the generated SQL failed to execute (see error above).")
                else:
                    order_sensitive = "order by" in st.session_state.gold_sql_used.lower()
                    verdict = compare_results(
                        st.session_state.query_cols, st.session_state.query_results,
                        st.session_state.gold_cols, st.session_state.gold_rows,
                        order_sensitive=order_sensitive,
                    )

                    if verdict["match"]:
                        n = len(st.session_state.query_results) if isinstance(st.session_state.query_results, list) else 0
                        st.success(f"Execution match — both queries returned the same {n} row(s).")
                    else:
                        st.error("Execution mismatch")
                        if not verdict["columns_match"]:
                            st.caption(
                                f"Columns differ — generated: `{st.session_state.query_cols}` "
                                f"vs gold: `{st.session_state.gold_cols}`"
                            )
                        diff_col1, diff_col2 = st.columns(2)
                        with diff_col1:
                            st.caption(f"Rows only in generated output ({len(verdict['only_in_generated'])})")
                            if verdict["only_in_generated"]:
                                st.dataframe(
                                    pd.DataFrame(verdict["only_in_generated"][:10], columns=st.session_state.query_cols),
                                    use_container_width=True
                                )
                        with diff_col2:
                            st.caption(f"Rows only in gold output ({len(verdict['only_in_gold'])})")
                            if verdict["only_in_gold"]:
                                st.dataframe(
                                    pd.DataFrame(verdict["only_in_gold"][:10], columns=st.session_state.gold_cols),
                                    use_container_width=True
                                )

    # TAB 2: SCHEMA VISUALIZER
    with tab2:
        st.subheader("Schema Structure")
        st.caption("Visual representation of the parsed DDL from the Query tab.")
        render_schema_cards(st.session_state.parsed_schema)

        if st.session_state.parsed_schema:
            st.divider()
            st.subheader("Entity-Relationship Diagram")
            st.caption(
                "Built directly from your DDL — always accurate, independent of "
                "whatever the AI model draws for a given query."
            )
            er_diagram_code = build_er_diagram(st.session_state.parsed_schema)
            if er_diagram_code:
                render_mermaid(er_diagram_code, height=420)
            else:
                st.info("No relationships to diagram yet.")


if __name__ == "__main__":
    main()