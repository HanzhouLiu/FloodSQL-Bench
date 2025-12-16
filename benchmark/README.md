### 🔎 Note

The `benchmark.jsonl` file contains the **older version** of the L4 questions (`triple_table_key_spatial`) (where some questions are mis-classified),  
and the reported results are evaluated on the **updated** L4 questions (`triple_table_key_spatial_updated`).
We recommend to use the `*.jsonl` under each question category or `bechmark_updated.jsonl`.

We welcome and encourage contributions from the community to further improve the question–SQL design.

### ⚠️ Known Limitations

- Some questions may not fully or accurately reflect their corresponding SQL.  
- Although all SQL queries are executable, certain queries may not be the *only* valid answer and may omit necessary conditions.  
- A subset of questions may be misclassified across difficulty levels.  
- The current dataset contains **443 question–SQL pairs**, which is relatively limited and will need to be expanded.
