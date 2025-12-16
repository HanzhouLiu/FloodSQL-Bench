import os
import json
#from openai import OpenAI
from litellm import completion, embedding
import numpy as np

import argparse

# ------------------------------
# Parse command line arguments
# ------------------------------
parser = argparse.ArgumentParser(description="Run RAG SQL benchmark")

parser.add_argument(
    "--model_name",
    type=str,
    default="huggingface/deepseek-ai/DeepSeek-V3.2",
    help="Name of the LLM model"
)

parser.add_argument(
    "--output_jsonl",
    type=str,
    default="results/DeepSeek-V3.2.jsonl",
    help="Path to output JSONL file"
)

args = parser.parse_args()

# =========================================================
# CONFIG
# =========================================================
# (MODIFIED) merged benchmark file
#INPUT_JSON = "benchmark/benchmark.jsonl"
INPUT_JSON = "benchmark/triple_table_key_spatial_updated/50_results.jsonl"

OUTPUT_JSONL = args.output_jsonl

DATA_DIR = "data"
METADATA_PATH = os.path.join(DATA_DIR, "metadata_parquet.json")

MODEL_NAME = args.model_name
EMBED_MODEL = "text-embedding-3-large"
# =========================================================
# HELPERS
# =========================================================
def embed(text: str):
    r = embedding(
        model=EMBED_MODEL,
        input=text
    )
    return np.array(r["data"][0]["embedding"])


def cosine(a, b):
    """Cosine similarity"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b) / denom)


def clean_sql(sql: str):
    """Remove Markdown SQL fences"""
    return sql.strip().replace("```sql", "").replace("```", "").strip()


def flatten_sql(sql: str) -> str:
    """Flatten SQL to one line for consistency"""
    return " ".join(sql.split())


# =========================================================
# LOAD METADATA
# =========================================================
def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# BUILD TABLE + COLUMN DESCRIPTION CORPUS
# =========================================================
def build_table_index(meta):
    """
    Build a text description for each table for embedding retrieval.
    """
    table_texts = {}

    for table, info in meta.items():
        if table == "_global":
            continue

        desc_parts = [table]

        # Combine column names + descriptions
        for col in info.get("schema", []):
            cname = col.get("column_name", "")
            cdesc = col.get("description", "")
            desc_parts.append(f"{cname}: {cdesc}")

        table_texts[table] = "\n".join(desc_parts)

    return table_texts


def build_column_index(meta):
    """
    Build column-level description index for fine-grained RAG.
    Returns: col_index[table] = [(column_name, description_text)]
    """
    col_index = {}

    for table, info in meta.items():
        if table == "_global":
            continue

        items = []
        for col in info.get("schema", []):
            cname = col.get("column_name", "")
            cdesc = col.get("description", "")
            text = f"{cname}: {cdesc}"
            items.append((cname, text))

        col_index[table] = items

    return col_index


# =========================================================
# RAG: TABLE RETRIEVAL (TOP_K)
# =========================================================
def retrieve_tables(question, table_index, top_k):
    q_emb = embed(question)
    scores = []

    for table, text in table_index.items():
        t_emb = embed(text)
        scores.append((table, cosine(q_emb, t_emb)))

    # Sort by relevance and take TOP_K
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_tables = [t for t, s in scores[:top_k]]

    return top_tables, scores[:top_k]


# =========================================================
# RAG: COLUMN RETRIEVAL (TOP 5 PER TABLE)
# =========================================================
def retrieve_columns(question, col_index, chosen_tables):
    q_emb = embed(question)
    result = {}

    for table in chosen_tables:
        cols = col_index[table]
        col_scores = []

        for cname, ctext in cols:
            cemb = embed(ctext)
            score = cosine(q_emb, cemb)
            col_scores.append((cname, score, ctext))

        col_scores = sorted(col_scores, key=lambda x: x[1], reverse=True)

        # Keep top 5 (or fewer if not enough)
        top_cols = col_scores[:5] if len(col_scores) >= 5 else col_scores
        result[table] = top_cols

    return result


# =========================================================
# BUILD METADATA PROMPT
# =========================================================
def build_metadata_prompt(meta, chosen_tables, chosen_columns):
    """
    Build the RAG prompt containing:
    - Selected tables
    - Selected columns
    - Global rules
    - Notes
    - Join rules (key-based & spatial)
    """
    lines = []

    # ================================
    # SELECTED TABLES
    # ================================
    lines.append("[TABLES SELECTED]")
    for t in chosen_tables:
        lines.append(f"- {t}")

    # ================================
    # SELECTED COLUMNS
    # ================================
    lines.append("\n[COLUMNS SELECTED]")
    for t, cols in chosen_columns.items():
        for cname, score, cdesc in cols:
            lines.append(f"- {t}.{cname}: {cdesc}")

    # ================================
    # JOIN RULES
    # ================================
    jr = meta["_global"].get("join_rules", {})

    # key-based direct
    direct_pairs = jr.get("key_based", {}).get("direct", [])
    lines.append("\n[JOIN RULES: KEY-BASED DIRECT]")
    for it in direct_pairs:
        p = it.get("pair", [])
        if len(p) == 2:
            lines.append(f"- {p[0]}  <->  {p[1]}")

    # key-based concat
    concat_pairs = jr.get("key_based", {}).get("concat", [])
    lines.append("\n[JOIN RULES: KEY-BASED CONCAT]")
    for it in concat_pairs:
        p = it.get("pair", [])
        if len(p) == 2:
            lines.append(f"- {p[0]}  <->  {p[1]}")

    # spatial point-polygon
    ppairs = jr.get("spatial", {}).get("point_polygon", [])
    lines.append("\n[JOIN RULES: SPATIAL POINT-POLYGON]")
    for it in ppairs:
        p = it.get("pair", [])
        if len(p) == 2:
            lines.append(f"- {p[0]}  <->  {p[1]}")

    # spatial polygon-polygon
    poly_pairs = jr.get("spatial", {}).get("polygon_polygon", [])
    lines.append("\n[JOIN RULES: SPATIAL POLYGON-POLYGON]")
    for it in poly_pairs:
        p = it.get("pair", [])
        if len(p) == 2:
            lines.append(f"- {p[0]}  <->  {p[1]}")

    # global rules
    rules = meta["_global"].get("rules", {})
    lines.append("\n[RULES]")
    for k, v in rules.items():
        lines.append(f"- {k}: {v}")

    # notes
    notes = meta["_global"].get("notes", [])
    lines.append("\n[NOTES]")
    for n in notes:
        lines.append(f"- {n}")

    # spatial notes
    sp = meta["_global"].get("spatial_function_notes", [])
    lines.append("\n[SPATIAL-NOTES]")
    for s in sp:
        lines.append(f"- {s}")

    result = "\n".join(lines)

    if len(result) > 80000:
        result = result[:80000] + "\n...[TRUNCATED]..."
        print("⚠ Metadata prompt truncated to 80000 characters.")
        
    return result


# =========================================================
# MAIN SQL GENERATION PIPELINE
# =========================================================
SYSTEM_PROMPT = """
You are an expert DuckDB SQL generator for the FloodSQL_Bench dataset.
Use only the tables and columns given in the metadata context.
Do NOT output any reasoning, explanation, or analysis.
Output only the final SQL query, with no comments and no semicolon.
Your output must contain SQL code only. 
Any natural language or reasoning is strictly forbidden.
"""


def generate_sql_rag_embed():
    meta = load_metadata(METADATA_PATH)

    # 1) indexing
    table_index = build_table_index(meta)
    col_index = build_column_index(meta)

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    fout = open(OUTPUT_JSONL, "w", encoding="utf-8")

    # 2) load merged benchmark
    qa_data = []
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        for line in f:
            qa_data.append(json.loads(line))

    # 3) iterate over all L0-L5 questions
    for item in qa_data:
        qid = item["id"]
        question = item["question"]
        # Extract level from id, e.g. "L4_0020" -> "L4"
        level = item["id"].split("_")[0]

        print(f"\n[{qid}] Processing question:\n{question}")

        # (MODIFIED) Determine TOP_K based on level
        if level.startswith("L0"):
            top_k = 3
        elif level.startswith("L1") or level.startswith("L2"):
            top_k = 4
        else:  # L3, L4, L5
            top_k = 5

        # STEP 1: retrieve tables
        chosen_tables, table_scores = retrieve_tables(question, table_index, top_k)

        # STEP 2: retrieve columns
        chosen_columns = retrieve_columns(question, col_index, chosen_tables)

        # STEP 3: build metadata prompt
        metadata_prompt = build_metadata_prompt(meta, chosen_tables, chosen_columns)

        user_prompt = f"""
Question:
{question}

Return only a single valid DuckDB SQL query.
"""

        # STEP 4: LLM
        try:
            resp = completion(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "system", "content": metadata_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            sql = clean_sql(resp["choices"][0]["message"]["content"])
            sql = flatten_sql(sql)


        except Exception as e:
            print(f"[ERROR] GPT failed on {qid}: {e}")
            sql = None

        record = {
            "id": qid,
            "question": question,
            "gt_sql": item.get("sql", None),
            "generated_sql": sql,
            "chosen_tables": chosen_tables,
            "chosen_columns": {
                t: [c for c, _, d in cols] for t, cols in chosen_columns.items()
            },
            "table_scores": table_scores,
        }

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()
    print(f"\n[DONE] Saved SQL to {OUTPUT_JSONL}")


if __name__ == "__main__":
    generate_sql_rag_embed()
