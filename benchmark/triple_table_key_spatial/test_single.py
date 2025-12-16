import duckdb
import os
import json
import signal
import time
from contextlib import contextmanager

# ==============================================
# Timeout handler
# ==============================================
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise TimeoutException(f"Query execution exceeded {seconds} seconds")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# ==============================================
# Register Parquet files
# ==============================================
def register_parquet_views(con, base_dir="data"):
    mapping = {
        "claims": "claims_tx_fl_la.parquet",
        "census_tracts": "census_tracts_tx_fl_la.parquet",
        "county": "county_tx_fl_la.parquet",
        "cre": "cre_tx_fl_la.parquet",
        "floodplain": "floodplain_tx_fl_la.parquet",
        "hospitals": "hospitals_tx_fl_la.parquet",
        "nri": "nri_tx_fl_la.parquet",
        "schools": "schools_tx_fl_la.parquet",
        "svi": "svi_tx_fl_la.parquet",
        "zcta": "zcta_tx_fl_la.parquet",
    }
    for view, fname in mapping.items():
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM '{path}'")

# ==============================================
# Pretty print helper
# ==============================================
def print_result(rows, limit=5):
    if not rows:
        print("  (no rows)")
        return
    if len(rows) == 1 and len(rows[0]) == 1:
        print("  ", rows[0][0])
        return
    for r in rows[:limit]:
        print("  ", " | ".join(str(x) for x in r))
    if len(rows) > limit:
        print(f"  ... ({len(rows)-limit} more rows)")

# ==============================================
# Run one query by ID
# ==============================================
def run_single_query(json_file, query_id, base_dir="data", timeout_seconds=30):
    con = duckdb.connect()
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    register_parquet_views(con, base_dir)

    with open(json_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    item = next((q for q in qa_data if q["id"] == query_id), None)
    if not item:
        print(f"Query ID {query_id} not found in {json_file}")
        return

    print("=" * 80)
    print(f"[{item['id']}] {item['question']}")
    print("-" * 80)
    print(item["sql"])
    print("=" * 80)

    try:
        start = time.time()
        with time_limit(timeout_seconds):
            rows = con.execute(item["sql"]).fetchall()
        elapsed = time.time() - start
        print(f"\n✅ Success ({len(rows)} rows, {elapsed:.3f}s)\n")
        print_result(rows)
    except TimeoutException as e:
        print(f"\n⏰ Timeout: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        con.close()

# ==============================================
# Example Usage
# ==============================================
if __name__ == "__main__":
    run_single_query(
        json_file="benchmark/triple_table_key_spatial/50.json",
        query_id="L4_0025",   # change this ID to the one you want
        base_dir="data",
        timeout_seconds=30,
    )
