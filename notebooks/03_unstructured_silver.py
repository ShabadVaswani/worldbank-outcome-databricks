# Databricks notebook source
# MAGIC %md
# MAGIC # 03_unstructured_silver — document text → features
# MAGIC
# MAGIC Turns each project's pre-approval document into numbers a model can use.
# MAGIC
# MAGIC | Feature family | Example | Intuition |
# MAGIC |---|---|---|
# MAGIC | Size | `n_words` | Very short documents (old memos) vs. full appraisals |
# MAGIC | Risk language (per 1,000 words) | `kw_capacity`, `kw_fragility`, `kw_delay` | How much the authors themselves worried about known failure causes |
# MAGIC | Style | `flesch_reading_ease`, `avg_sentence_words`, `pct_numeric_tokens` | Clarity and how quantified the plan is |
# MAGIC | Meaning | 384-dim sentence embedding (reduced to 20 dims in `04_gold`) | What the project is *about*, beyond keywords |
# MAGIC
# MAGIC Outputs: `silver.doc_text_features`, `silver.doc_embeddings`. **Needs:** `01b_ingest_docs`. **Attach to:** `wb-ml-single`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings
# MAGIC
# MAGIC **What:** keyword lists, which part of each document to embed, and the embedding model.
# MAGIC
# MAGIC **Why:**
# MAGIC - **Keyword families** are small, readable lists so every feature is explainable (good for SHAP and interviews).
# MAGIC - **Embedding window:** we skip the first 300 words (cover page, abbreviations, currency tables) and embed the next 8 chunks of 200 words — the strategic context and project description. Embedding whole 100-page documents on a 4-core CPU would take hours for little gain.
# MAGIC - **Model:** `all-MiniLM-L6-v2` — small (22M parameters), fast on CPU, 384-dim output, and already supported by the preinstalled `sentence-transformers`.

# COMMAND ----------

import json, re, time
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

CATALOG = "dbw_wb_outcome"
SKIP_WORDS, CHUNK_WORDS, N_CHUNKS = 300, 200, 8
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

KEYWORDS = {
    "kw_risk": [r"\brisks?\b"],
    "kw_uncertainty": [r"\buncertain(ty|ties)?\b", r"\bmay not\b", r"\bunpredictab\w*"],
    "kw_capacity": [r"\b(weak|limited|low|lack of|insufficient) (institutional |implementation )?capacity\b",
                    r"\bcapacity constraints?\b"],
    "kw_fragility": [r"\bconflict\w*\b", r"\bfragil\w*\b", r"\binsecurity\b", r"\bviolence\b"],
    "kw_governance": [r"\bcorrupt\w*\b", r"\bfraud\w*\b", r"\bgovernance\b"],
    "kw_delay": [r"\bdelay\w*\b"],
    "kw_procurement": [r"\bprocurement\b"],
    "kw_mitigation": [r"\bmitigat\w*\b"],
    "kw_lessons": [r"\blessons?\b"],
    "kw_monitoring": [r"\bmonitoring\b", r"\bbaselines?\b", r"\bindicators?\b"],
    "kw_objective": [r"\bobjectives?\b"],
    "kw_reform": [r"\breforms?\b"],
}
KW_RE = {k: re.compile("|".join(v), re.I) for k, v in KEYWORDS.items()}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the downloaded documents
# MAGIC
# MAGIC **What:** lists successfully downloaded files from `bronze.doc_downloads` with their manifest info (doc type, doc date, approval date).
# MAGIC
# MAGIC **Why:** the download log is the source of truth for which text exists; projects whose download failed simply get no text features (XGBoost handles missing values natively).

# COMMAND ----------

files = (spark.table(f"{CATALOG}.bronze.doc_downloads").filter("status in ('ok','exists')")
         .join(spark.table(f"{CATALOG}.bronze.doc_manifest"), ["project_id", "doc_id"])
         .select("project_id", "doc_id", "path", "doc_type", "doc_date", "approval_date")
         .toPandas())
print("documents to process:", len(files))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Clean text and compute hand-crafted features
# MAGIC
# MAGIC **What:** for each file: decode, drop table-of-contents dot leaders and page-number lines, collapse whitespace; then compute size, keyword rates per 1,000 words, sentence length, an approximate Flesch reading-ease score, and the share of numeric tokens. Also keeps the embedding window text.
# MAGIC
# MAGIC **Why:**
# MAGIC - **Rates, not counts:** a 200-page PAD naturally says "risk" more often than a 10-page memo; dividing by length makes documents comparable.
# MAGIC - **Flesch reading ease** = 206.835 − 1.015 × (words/sentence) − 84.6 × (syllables/word); higher = easier to read. Syllables are estimated by counting vowel groups — rough, but consistent across documents, which is what matters for a feature.
# MAGIC - Old documents are OCR'd scans with noise; cleaning keeps the noise from dominating the counts.

# COMMAND ----------

DOT_LEADER = re.compile(r"\.{3,}|_{3,}|-{3,}")
PAGE_LINE = re.compile(r"(?m)^\s*(page\s*)?\d{1,3}\s*$", re.I)
WS = re.compile(r"\s+")
SENT = re.compile(r"[.!?]+(?=\s|$)")
VOWELS = re.compile(r"[aeiouy]+")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")

def syllables(word):
    return max(1, len(VOWELS.findall(word.lower())))

def process(path):
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8", errors="replace")
    text = WS.sub(" ", PAGE_LINE.sub(" ", DOT_LEADER.sub(" ", raw))).strip()
    tokens = text.split(" ")
    n = len(tokens)
    alpha = WORD.findall(text[:200_000])
    sample_words = alpha[:20_000]
    n_sent = max(1, len(SENT.findall(text[:200_000])))
    feats = {
        "n_words": n,
        "n_chars": len(text),
        "avg_sentence_words": len(alpha) / n_sent if alpha else None,
        "pct_numeric_tokens": sum(t[:1].isdigit() for t in tokens) / n if n else None,
    }
    if sample_words:
        wps = len(sample_words) / max(1, len(SENT.findall(" ".join(sample_words))))
        spw = sum(syllables(w) for w in sample_words) / len(sample_words)
        feats["flesch_reading_ease"] = 206.835 - 1.015 * wps - 84.6 * spw
    per_k = 1000 / n if n else 0
    for k, rx in KW_RE.items():
        feats[k] = len(rx.findall(text)) * per_k
    window = tokens[SKIP_WORDS:SKIP_WORDS + CHUNK_WORDS * N_CHUNKS] or tokens[:CHUNK_WORDS * N_CHUNKS]
    chunks = [" ".join(window[i:i + CHUNK_WORDS]) for i in range(0, len(window), CHUNK_WORDS)]
    return feats, chunks

t0 = time.time()
feat_rows, chunk_rows = [], []
for i, r in enumerate(files.itertuples(index=False), 1):
    try:
        feats, chunks = process(r.path)
    except Exception as e:
        print("skip", r.path, type(e).__name__)
        continue
    feats.update(project_id=r.project_id, doc_id=r.doc_id, doc_type=r.doc_type,
                 doc_lead_days=(pd.Timestamp(r.approval_date) - pd.Timestamp(r.doc_date)).days)
    feat_rows.append(feats)
    chunk_rows.extend((r.project_id, j, c) for j, c in enumerate(chunks))
    if i % 1000 == 0:
        print(f"{i}/{len(files)} processed ({time.time() - t0:.0f}s)")

text_feats = pd.DataFrame(feat_rows)
chunks_df = pd.DataFrame(chunk_rows, columns=["project_id", "chunk_idx", "chunk_text"])
print("features:", text_feats.shape, "| chunks:", len(chunks_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save text features and chunks
# MAGIC
# MAGIC **What:** writes one row per project to `silver.doc_text_features` and the embedding-window chunks to `silver.doc_chunks`.
# MAGIC
# MAGIC **Why:** features are the model input; chunks are kept so the embedding step (and an optional search/RAG demo) can be re-run without re-reading 7,000 files.

# COMMAND ----------

num_cols = [c for c in text_feats.columns if c not in ("project_id", "doc_id", "doc_type")]
text_feats[num_cols] = text_feats[num_cols].astype("float64")
spark.createDataFrame(text_feats).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.silver.doc_text_features")
spark.createDataFrame(chunks_df).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.silver.doc_chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Embeddings
# MAGIC
# MAGIC **What:** encodes chunks with MiniLM (normalized vectors), then averages each project's chunk vectors into one 384-dim **document embedding**, saved to `silver.doc_embeddings` together with the `doc_id` and model name it came from.
# MAGIC
# MAGIC **Incremental:** a project is re-embedded only if it's new, its chosen document changed, or the model changed — everything else is reused from the existing table. The first full run takes ~60 minutes on this 4-core CPU; later runs only embed new projects (minutes). Set the `force_reembed` widget to `true` to rebuild everything (e.g. after changing the chunking).
# MAGIC
# MAGIC **Why:** an embedding places texts with similar meaning close together, so the model can learn from *what kind of project* this is (e.g. "rural roads in a post-conflict state") — something keyword counts miss. Mean-pooling chunks is the simplest robust way to get one vector per document.

# COMMAND ----------

import torch
from sentence_transformers import SentenceTransformer

dbutils.widgets.dropdown("force_reembed", "false", ["false", "true"])
FORCE = dbutils.widgets.get("force_reembed") == "true"
emb_table = f"{CATALOG}.silver.doc_embeddings"
current = text_feats[["project_id", "doc_id"]]

old = None
if not FORCE and spark.catalog.tableExists(emb_table):
    old = spark.table(emb_table).toPandas()
    if "doc_id" in old.columns:
        old = old[old["model"] == EMBED_MODEL].merge(current, on=["project_id", "doc_id"])
    else:  # table from the first pipeline version (no doc_id/model columns): same model, same documents
        old = old.merge(current, on="project_id")
todo_ids = set(current["project_id"]) - (set(old["project_id"]) if old is not None else set())
todo_chunks = chunks_df[chunks_df["project_id"].isin(todo_ids)]
print(f"reusing {0 if old is None else len(old)} embeddings, embedding {len(todo_ids)} projects ({len(todo_chunks)} chunks)")

new = pd.DataFrame(columns=["project_id", "embedding"])
if len(todo_chunks):
    torch.set_num_threads(4)
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    t0 = time.time()
    vecs = model.encode(todo_chunks["chunk_text"].tolist(), batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
    print(f"encoded {len(vecs)} chunks in {time.time() - t0:.0f}s")
    tc = todo_chunks.assign(vec=list(vecs))
    doc_vecs = tc.groupby("project_id")["vec"].apply(lambda v: np.mean(np.stack(v.values), axis=0))
    new = pd.DataFrame({"project_id": doc_vecs.index,
                        "embedding": [v.astype("float32").tolist() for v in doc_vecs.values]})

emb = pd.concat([old[["project_id", "embedding"]] if old is not None else None, new], ignore_index=True)
emb = emb.merge(current, on="project_id").assign(model=EMBED_MODEL)
emb["embedding"] = emb["embedding"].apply(lambda v: [float(x) for x in v])
spark.createDataFrame(emb[["project_id", "doc_id", "model", "embedding"]],
                      "project_id string, doc_id string, model string, embedding array<float>") \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(emb_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Checks and summary
# MAGIC
# MAGIC **What:** row counts, embedding dimension, and feature averages by document type.
# MAGIC
# MAGIC **Why:** confirms every processed document got both feature sets, and shows whether feature values differ wildly by document type (if they do, `doc_type` must be a feature so the model doesn't confuse "short memo" with "risky project").

# COMMAND ----------

tf = spark.table(f"{CATALOG}.silver.doc_text_features")
display(tf.groupBy("doc_type").agg(F.count("*").alias("docs"), F.round(F.avg("n_words")).alias("avg_words"),
                                   F.round(F.avg("kw_risk"), 2).alias("avg_kw_risk"),
                                   F.round(F.avg("flesch_reading_ease"), 1).alias("avg_flesch")))
summary = {
    "docs_with_features": tf.count(),
    "docs_with_embeddings": spark.table(f"{CATALOG}.silver.doc_embeddings").count(),
    "embedding_dim": int(len(emb["embedding"].iloc[0])),
    "chunks": len(chunks_df),
}
print(summary)
dbutils.notebook.exit(json.dumps(summary))
