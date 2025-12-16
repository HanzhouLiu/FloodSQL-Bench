import duckdb
import glob
import os

def export_schema(base_path, output_file):
    con = duckdb.connect()
    parquet_files = glob.glob(os.path.join(base_path, "*.parquet"))

    with open(output_file, "w", encoding="utf-8") as f:
        for fpath in parquet_files:
            fname = os.path.basename(fpath)
            f.write(f"\n===== {fname} =====\n")

            # 获取 schema
            df = con.execute(f"DESCRIBE SELECT * FROM parquet_scan('{fpath}')").fetchdf()
            f.write(df.to_string())
            f.write("\n")

    print(f"Schema summary written to {output_file}")

if __name__ == "__main__":
    base_path = "FloodSQL_Bench/data"
    output_file = "FloodSQL_Bench/schema_summary.txt"
    export_schema(base_path, output_file)
