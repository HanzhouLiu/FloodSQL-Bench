# FloodSQL-Bench: A Retrieval-Augmented Benchmark for Geospatially-Grounded Text-to-SQL 
[![arXiv](https://img.shields.io/badge/arXiv-2512.12084-b31b1b.svg)](https://arxiv.org/abs/2512.12084) [![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-FloodSQL--Bench-yellow)](https://huggingface.co/HanzhouLiu/FloodSQL-Bench)

**Authors**  
Hanzhou Liu · Kai Yin · Zhitong Chen · Chenyue Liu · Ali Mostafavi

**Abstract**  
Existing Text-to-SQL benchmarks primarily focus on single-table queries or limited joins in general-purpose domains, and thus fail to reflect the complexity of domain-specific, multitable and geospatial reasoning, To address this limitation, we introduce FLOODSQL-BENCH, a geospatially grounded benchmark for the flood management domain that integrates heterogeneous datasets through key-based, spatial, and hybrid joins. The benchmark captures realistic flood-related information needs by combining social, infrastructural, and hazard data layers. We systematically evaluate recent large language models with the same retrieval-augmented generation settings and measure their performance across difficulty tiers. By providing a unified, open benchmark grounded in real-world disaster management data, FLOODSQL-BENCH establishes a practical testbed for advancing Text-to-SQL research in high-stakes application domains.

---

## 📂 Project Structure

- **data/** → Parquet datasets and metadata
- **benchmark/** → Generated Question-SQL pairs
- **scripts/** → Utilities (preview scripts, metadata generation, SQL tests)  
- **README.md** → Detailed dataset/benchmark documentation  

---

## 🚀 Quick Start

Install the recommended dependencies following `requirements.txt`.  
If you encounter any version conflicts or installation issues, please open an issue.

```bash
# create and activate conda environment
conda create -n floodsql python=3.10 -y
conda activate floodsql
pip install duckdb pandas geopandas sqlparse shapely pyarrow
conda install -c conda-forge libspatialite

# preview a dataset
python scripts/preview/census_tracts.py

# generate metadata
python scripts/generate_metadata.py

# debug benchmark without llm agents
python benchmark/single_table/run_50.py

# run and evaluate llm agent on FloodSQL
# you might want to modify the input/output file name/path in the codes
python scripts/run_and_eval/run.py
python scripts/run_and_eval/eval.py
```
## 📥 Dataset Download

The FloodSQL-Bench dataset is publicly available on Hugging Face:

**https://huggingface.co/datasets/HanzhouLiu/FloodSQL-Bench**

You can download the entire dataset via the Hugging Face web interface.
data/
└── *.parquet # All dataset files are stored in Parquet format

```bash
# You might want to setup your hf tokens before running the following script.
python scripts/download_hf.py
```
## 📊 Benchmark
./benchmark # All Question-SQL pairs are stored in this directory.
See the detailed benchmark description in the [Benchmark README](benchmark/README.md).

## 🗃️ Metadata
Metadata Preview
```
{
  "claims": {
      "schema": [
          {"column_name": ..., "description": ...},
          ...
      ],
      "other table-specific info..."
  },
  "census_tracts": {...},
  "nri": {...},
  "svi": {...},
  "cre": {...},
  "hospitals": {...},
  "schools": {...},
  "county": {...},
  "zcta": {...},
  "floodplain": {...},

  "_global": {
      "join_rules": {...},
      "rules": {...},
      "notes": [...],
      "triple_table_notes": [...],
      "spatial_function_notes": [...],
      "basic_function_notes": [...]
  }
}
```

## 🔗 🤖 Cross-Table RAG 
```
We use text embeddings and cosine similarity to select multiple Top-K candidates 
at the Table, and then at the Column level.
The following messages will be fed into the LLM agent:
messages = [
  {"role": "system", "content": SYSTEM_PROMPT},    # ← "You are a Text-to-SQL expert"
  {"role": "system", "content": metadata_prompt},  # ← candidates + global rules
  {"role": "user", "content": user_prompt},        # ← Question / User query
]
```


### ⚠️ Known Limitations
Dataset Quality & Coverage
- Some questions may not fully or accurately reflect their corresponding SQL.  
- Although all SQL queries are executable, certain queries may not be the *only* valid answer and may omit necessary conditions.  
- A subset of questions may be misclassified across difficulty levels.  
- The current dataset contains **443 question–SQL pairs**, which is relatively limited and will need to be expanded.
- Since our codes are based on LLM-Agent API, the returned null results might be caused by multiple reasons. These null values might to some extent affect the metric score.
- There might be some duplicate questions.

LLM-Agent Execution Behavior

- Our pipeline uses an LLM-Agent–based system. As a result, some returned null results may originate from model hallucination, API timeout, agent routing errors, or incomplete tool invocation. We have not investigated the root cause. Any finding about it would be appreciated.

- These null outputs may introduce noise in evaluation metrics. Deepseek-r1-1.5b is such an example. We label the models with such issues with a symbol *. 

If you would like to use our benchmark in your work, please cite this paper as follows,
```
@misc{liu2025floodsqlbenchretrievalaugmentedbenchmarkgeospatiallygrounded,
      title={FloodSQL-Bench: A Retrieval-Augmented Benchmark for Geospatially-Grounded Text-to-SQL}, 
      author={Hanzhou Liu and Kai Yin and Zhitong Chen and Chenyue Liu and Ali Mostafavi},
      year={2025},
      eprint={2512.12084},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2512.12084}, 
}
```
Thanks for you interest in FloodSQL-Bench! 😊