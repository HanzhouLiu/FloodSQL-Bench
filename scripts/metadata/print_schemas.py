import json

with open("db/metadata_enhanced.json") as f:
    metadata = json.load(f)

target_tables = [
    "claims_tx_fl_la",
    "census_tracts_tx_fl_la",
    "county_tx_fl_la"
]

# print schema
for t in target_tables:
    if t not in metadata:
        print(f"Table {t} not in metadata")
        continue
    cols = [col["column_name"] for col in metadata[t].get("schema", [])]
    print(f"\n {t} ({len(cols)} 列)")
    print(", ".join(cols))
