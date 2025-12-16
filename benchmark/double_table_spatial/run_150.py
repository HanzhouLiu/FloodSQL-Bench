import duckdb
import os
import json
import time
import signal
from contextlib import contextmanager

# ============================================================
# Timeout Handler
# ============================================================
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager to enforce a time limit on code execution."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Query execution exceeded {seconds} second timeout")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# ============================================================
# Register parquet files as DuckDB views
# ============================================================
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
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM '{fpath}'")
        else:
            print(f"Missing parquet: {fpath}")


# ============================================================
# Helper: formatted output for console
# ============================================================
def format_result(rows, limit=5):
    """Pretty-print query results depending on shape."""
    if not rows:
        return "  (no rows)\n"

    # Single scalar value
    if len(rows) == 1 and len(rows[0]) == 1:
        return f"  {rows[0][0]}\n"

    # Multi-column formatted table
    cols = len(rows[0])
    col_widths = [max(len(str(row[i])) for row in rows[:limit]) for i in range(cols)]
    lines = []
    for i, row in enumerate(rows[:limit]):
        formatted = "  " + " | ".join(
            str(row[j]).ljust(col_widths[j]) for j in range(cols)
        )
        lines.append(formatted)
    if len(rows) > limit:
        lines.append(f"  ... ({len(rows) - limit} more rows)")
    return "\n".join(lines) + "\n"


# ============================================================
# Run gold queries from JSON and export JSONL results
# ============================================================
def run_gold_queries(
    json_file,
    base_dir="data",
    limit_output=5,
    timeout_seconds=30,
    txt_out="benchmark/seed_qa_25_test.txt",
    jsonl_out="benchmark/seed_qa_25_results.jsonl",
    error_ids_out="benchmark/error_ids.txt",
):
    con = duckdb.connect()
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")

    register_parquet_views(con, base_dir)

    with open(json_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    os.makedirs(os.path.dirname(txt_out), exist_ok=True)
    
    # Track error IDs and details
    error_ids = []
    error_details = []
    timeout_ids = []
    
    with open(txt_out, "w", encoding="utf-8") as fout, open(
        jsonl_out, "w", encoding="utf-8"
    ) as jout:
        for i, item in enumerate(qa_data, start=1):
            qid = item.get("id", f"Q{i}")
            question = item.get("question")
            sql = item.get("sql")

            header = "=" * 80 + "\n"
            header += f"[{qid}] {question}\n"
            header += "-" * 80 + "\n"
            header += sql + "\n\n"

            print(header)
            fout.write(header)

            record = {"id": qid, "question": question, "sql": sql}

            try:
                start = time.time()
                
                # Execute with timeout
                with time_limit(timeout_seconds):
                    result = con.execute(sql).fetchall()
                
                elapsed = time.time() - start

                success_msg = f"Success ({len(result)} rows, {elapsed:.3f} sec)\n"
                print(success_msg)
                fout.write(success_msg)

                formatted = format_result(result, limit_output)
                print(formatted)
                fout.write(formatted)

                record["elapsed"] = round(elapsed, 3)
                record["row_count"] = len(result)
                record["result"] = result

            except TimeoutException as e:
                err_msg = f"Timeout Error: Query exceeded {timeout_seconds} seconds\n"
                print(err_msg)
                fout.write(err_msg)
                record["error"] = f"Timeout: exceeded {timeout_seconds}s"
                record["error_type"] = "timeout"
                
                # Track timeout separately
                error_ids.append(qid)
                timeout_ids.append(qid)
                error_details.append({
                    "id": qid,
                    "question": question,
                    "error": f"Timeout: Query exceeded {timeout_seconds} seconds",
                    "error_type": "timeout"
                })

            except Exception as e:
                err_str = str(e)
                
                # Treat "Query interrupted" as timeout
                if "Query interrupted" in err_str or "interrupted" in err_str.lower():
                    err_msg = f"Timeout Error: Query interrupted (exceeded {timeout_seconds} seconds)\n"
                    print(err_msg)
                    fout.write(err_msg)
                    record["error"] = f"Timeout: exceeded {timeout_seconds}s"
                    record["error_type"] = "timeout"
                    
                    # Track timeout
                    error_ids.append(qid)
                    timeout_ids.append(qid)
                    error_details.append({
                        "id": qid,
                        "question": question,
                        "error": f"Timeout: Query interrupted after {timeout_seconds} seconds",
                        "error_type": "timeout"
                    })
                else:
                    # Regular error
                    err_msg = f"Error while executing: {e}\n"
                    print(err_msg)
                    fout.write(err_msg)
                    record["error"] = err_str
                    
                    # Track error ID and details
                    error_ids.append(qid)
                    error_details.append({
                        "id": qid,
                        "question": question,
                        "error": err_str,
                        "error_type": "execution_error"
                    })

            jout.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    con.close()
    
    # Write error IDs to file
    with open(error_ids_out, "w", encoding="utf-8") as err_file:
        if error_ids:
            err_file.write(f"Total errors: {len(error_ids)} out of {len(qa_data)} queries\n")
            err_file.write(f"Error rate: {len(error_ids)/len(qa_data)*100:.2f}%\n")
            err_file.write(f"Timeout errors: {len(timeout_ids)}\n")
            err_file.write(f"Execution errors: {len(error_ids) - len(timeout_ids)}\n")
            err_file.write("=" * 80 + "\n\n")
            
            # Write just the IDs (one per line)
            err_file.write("Error IDs:\n")
            err_file.write("-" * 80 + "\n")
            for err_id in error_ids:
                err_file.write(f"{err_id}\n")
            
            # Write timeout IDs separately
            if timeout_ids:
                err_file.write("\n" + "=" * 80 + "\n")
                err_file.write("Timeout IDs (exceeded 30s):\n")
                err_file.write("-" * 80 + "\n")
                for tid in timeout_ids:
                    err_file.write(f"{tid}\n")
            
            # Write detailed error information
            err_file.write("\n" + "=" * 80 + "\n")
            err_file.write("Detailed Error Information:\n")
            err_file.write("=" * 80 + "\n\n")
            for detail in error_details:
                err_file.write(f"ID: {detail['id']}\n")
                err_file.write(f"Question: {detail['question']}\n")
                err_file.write(f"Error Type: {detail.get('error_type', 'unknown')}\n")
                err_file.write(f"Error: {detail['error']}\n")
                err_file.write("-" * 80 + "\n\n")
        else:
            err_file.write("No errors found. All queries executed successfully!\n")
    
    print(f"\n[DONE] Results saved to:")
    print(f" - {txt_out}")
    print(f" - {jsonl_out}")
    print(f" - {error_ids_out}")
    
    if error_ids:
        print(f"\n[SUMMARY] {len(error_ids)} errors out of {len(qa_data)} queries ({len(error_ids)/len(qa_data)*100:.1f}%)")
        print(f"  Timeout errors: {len(timeout_ids)}")
        print(f"  Execution errors: {len(error_ids) - len(timeout_ids)}")
        print(f"\nFailed query IDs: {', '.join(error_ids[:10])}" + (" ..." if len(error_ids) > 10 else ""))
    else:
        print(f"\n[SUMMARY] All {len(qa_data)} queries executed successfully!")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    json_file = "benchmark/double_table_spatial/150.json"
    run_gold_queries(
        json_file,
        base_dir="data",
        timeout_seconds=30,
        txt_out="benchmark/double_table_spatial/150_test.txt",
        jsonl_out="benchmark/double_table_spatial/150_results.jsonl",
        error_ids_out="benchmark/double_table_spatial/150_error_ids.txt",
    )