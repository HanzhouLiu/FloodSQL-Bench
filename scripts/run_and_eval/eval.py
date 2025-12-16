import json
import numpy as np
from litellm import embedding
import os
from tqdm import tqdm
from collections import defaultdict
import time

# ===== CONFIG =====
INPUT = "results/L4_Updated/gemma-2-9b-it.jsonl"  # Path to input JSONL file

OPENAI_EMBED_MODEL = "text-embedding-3-large"
JINA_EMBED_MODEL = "jina_ai/jina-embeddings-v3"

BATCH = 32  # batch size for embedding

# ===== HELPERS =====
def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b) / denom)


def batch_embed(texts, model_name, max_retry=3):
    """Batch embedding with retry"""
    for attempt in range(max_retry):
        try:
            r = embedding(model=model_name, input=texts)
            return [np.array(x["embedding"]) for x in r["data"]]
        except Exception as e:
            if attempt == max_retry - 1:
                raise e
            time.sleep(1.2)  # wait then retry


# ===== MAIN =====
def main():
    data = []
    with open(INPUT, "r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)

            gt = item.get("gt_sql") or "[EMPTY_GT_SQL]"
            gen = item.get("generated_sql") or "[EMPTY_GEN_SQL]"

            item["gt_sql"] = gt
            item["generated_sql"] = gen

            data.append(item)


    print(f"Loaded {len(data)} rows.")

    # Collect SQLs
    gt_sqls = [x["gt_sql"] for x in data]
    gen_sqls = [x["generated_sql"] for x in data]

    print("\nEmbedding OpenAI...")
    openai_gt_emb = []
    openai_gen_emb = []

    for i in tqdm(range(0, len(data), BATCH)):
        gt_batch = gt_sqls[i : i+BATCH]
        gen_batch = gen_sqls[i : i+BATCH]

        gt_vecs = batch_embed(gt_batch, OPENAI_EMBED_MODEL)
        gen_vecs = batch_embed(gen_batch, OPENAI_EMBED_MODEL)

        openai_gt_emb.extend(gt_vecs)
        openai_gen_emb.extend(gen_vecs)

    print("\nEmbedding Jina...")
    jina_gt_emb = []
    jina_gen_emb = []

    for i in tqdm(range(0, len(data), BATCH)):
        gt_batch = gt_sqls[i : i+BATCH]
        gen_batch = gen_sqls[i : i+BATCH]

        gt_vecs = batch_embed(gt_batch, JINA_EMBED_MODEL)
        gen_vecs = batch_embed(gen_batch, JINA_EMBED_MODEL)

        jina_gt_emb.extend(gt_vecs)
        jina_gen_emb.extend(gen_vecs)

    # ===== SCORING =====
    openai_scores = defaultdict(list)
    jina_scores = defaultdict(list)

    for item, gto, geno, gtj, genj in zip(data, openai_gt_emb, openai_gen_emb,
                                          jina_gt_emb, jina_gen_emb):
        level = item["id"].split("_")[0]
        openai_scores[level].append(cosine(gto, geno))
        jina_scores[level].append(cosine(gtj, genj))

    print("\n===== AVERAGE COSINE SIMILARITIES =====\n")

    for lv in ["L0", "L1", "L2", "L3", "L4", "L5"]:
        avg_o = np.mean(openai_scores[lv]) if openai_scores[lv] else None
        avg_j = np.mean(jina_scores[lv]) if jina_scores[lv] else None
        print(f"{lv}: OpenAI={avg_o}, Jina={avg_j}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
