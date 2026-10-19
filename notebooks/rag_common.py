# Databricks notebook source
# MAGIC %md
# MAGIC # rag_common — shared settings and the `search()` helper for the RAG notebooks
# MAGIC
# MAGIC **RAG** (retrieval-augmented generation) = before a language model answers, we *retrieve* the most relevant document passages ("chunks") and give them to the model as evidence.
# MAGIC
# MAGIC This small notebook holds what `09_rag_index` (builds the index), `10_rag_assistant`, `11_rag_evaluation` and `12_flagged_project_notes` all need:
# MAGIC
# MAGIC | Item | Purpose |
# MAGIC |---|---|
# MAGIC | constants | table, index, endpoint and model names in one place |
# MAGIC | `embed_texts()` | turns text into unit-length vectors with the `databricks-gte-large-en` embedding endpoint (batched, with retries) |
# MAGIC | `chat()` | calls a chat model endpoint and returns the final text, token usage and latency (handles reasoning models) |
# MAGIC | `search(query, k, filters, mode)` | returns the top-k chunks as a pandas DataFrame, from either backend |
# MAGIC
# MAGIC **Two interchangeable backends** (`VECTOR_BACKEND`):
# MAGIC - `"vector_search"` — Databricks Vector Search (also called Mosaic AI / AI Search): a managed service that serves an index over a Delta table. Supports **hybrid search** (vector similarity + keyword matching, results merged). Bills about 4 DBU/hour while the index exists.
# MAGIC - `"faiss"` — FAISS (Facebook AI Similarity Search), an open-source library; the index is a file in a Unity Catalog volume and is searched in this notebook's memory. Free to run, but **vector-only** (no keyword half).
# MAGIC
# MAGIC Use it with `%run ./rag_common` (after `%pip install` of the two client libraries). It only *reads*; it never rebuilds anything. If Vector Search is selected but the endpoint or index is missing or not ready, it **falls back to FAISS automatically** (see the last section).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings
# MAGIC
# MAGIC **What:** names of every object the RAG notebooks share, plus the backend choice.
# MAGIC
# MAGIC **Why:** one source of truth — if the index or model name changes, it changes here only. A calling notebook can set `VECTOR_BACKEND` *before* `%run ./rag_common` (for example from a widget); otherwise the default is Vector Search.

# COMMAND ----------

import json, time, random
import numpy as np
import pandas as pd

CATALOG = "dbw_wb_outcome"
CHUNKS_TABLE = f"{CATALOG}.silver.rag_chunks"          # one row per chunk, with its stored embedding
VS_ENDPOINT = globals().get("VS_ENDPOINT", "wb-rag")    # Vector Search endpoint (the serving compute); a caller may override
VS_INDEX = globals().get("VS_INDEX", f"{CATALOG}.gold.rag_chunks_index")   # Delta Sync index on CHUNKS_TABLE
EMBED_ENDPOINT = "databricks-gte-large-en"              # Foundation Model API embedding endpoint
EMBED_DIM = 1024                                        # measured: GTE-large-en v1.5 returns 1,024 numbers
ANSWER_MODEL = "databricks-qwen35-122b-a10b"            # answer model for 10 and 12 (a reasoning model)
JUDGE_MODEL = "databricks-gpt-oss-120b"                 # judge + eval-question generator in 11 (different model family)
FAISS_VOLUME = f"{CATALOG}.gold.rag_index"
FAISS_DIR = f"/Volumes/{CATALOG}/gold/rag_index/faiss"
RETURN_COLUMNS = ["chunk_id", "project_id", "doc_id", "doc_type", "project_name", "country", "region", "sector",
                  "approval_fy", "section", "chunk_index", "is_active", "risk_band", "risk_decile", "chunk_text"]
VECTOR_BACKEND = globals().get("VECTOR_BACKEND", "vector_search")
assert VECTOR_BACKEND in ("vector_search", "faiss"), VECTOR_BACKEND

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calling Databricks model endpoints: `embed_texts()` and `chat()`
# MAGIC
# MAGIC **What:**
# MAGIC - `_invoke()` posts JSON to a serving endpoint over HTTPS with the notebook's own credentials, an explicit timeout, and **exponential backoff** on throttling or transient errors (wait ~0.5 s, 1 s, 2 s, 4 s … up to 30 s, plus random jitter).
# MAGIC - `embed_texts(texts)` sends at most 150 texts per request to `databricks-gte-large-en` and returns **unit-length** float32 vectors (one row per text).
# MAGIC - `chat(messages, model, ...)` calls a chat model and returns the final answer text, any reasoning text, token usage and latency. It handles **both response shapes** the endpoints use: plain-string `content` (Llama, Qwen3-Next) and a *list of blocks* (`reasoning` block(s), then a `text` block) from reasoning models (gpt-oss, Qwen3.5).
# MAGIC
# MAGIC **Why:**
# MAGIC - Pay-per-token endpoints have a low per-workspace rate limit (HTTP 429 `REQUEST_LIMIT_EXCEEDED` appeared after one request per second in testing) and a 150-texts-per-request cap. The MLflow deployments client retries 429s silently with long waits, so a throttled run looked frozen; plain HTTP with our own short backoff keeps throttling visible and bounded.
# MAGIC - **GTE returns vectors of length ≈ 24, not 1.** Cosine similarity needs unit vectors, and Vector Search ranks self-managed vectors by L2 (straight-line) distance; for unit vectors both give the same ranking. So every vector — documents and queries — is scaled to length 1 here, in one place.
# MAGIC - Reasoning models spend output tokens "thinking" before answering; if `max_tokens` is too small they return only reasoning and no answer. `chat()` reports reasoning and answer separately so that cost and failures are visible.

# COMMAND ----------

import requests

_HOST = spark.conf.get("spark.databricks.workspaceUrl")
_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
EMBED_MAX_BATCH = 150                                   # endpoint limit: "exceeding 150 limit"
CALL_STATS = {"calls": 0, "retries": 0}                 # throttling visibility

def _invoke(endpoint, body, timeout=120, max_retries=10):
    for attempt in range(max_retries):
        try:
            r = requests.post(f"https://{_HOST}/serving-endpoints/{endpoint}/invocations",
                              headers={"Authorization": f"Bearer {_TOKEN}"}, json=body, timeout=timeout)
            CALL_STATS["calls"] += 1
            if r.status_code == 200:
                return r.json()
            retry = r.status_code == 429 or r.status_code >= 500
            err = f"HTTP {r.status_code}: {r.text[:300]}"
        except requests.exceptions.RequestException as e:
            retry, err = True, f"{type(e).__name__}: {str(e)[:200]}"
        if not retry or attempt == max_retries - 1:
            raise RuntimeError(f"{endpoint} failed: {err}")
        CALL_STATS["retries"] += 1
        time.sleep(min(30, 0.5 * 2 ** attempt) + random.random() * 0.5)

def embed_texts(texts, batch_size=EMBED_MAX_BATCH):
    out = []
    for i in range(0, len(texts), batch_size):
        resp = _invoke(EMBED_ENDPOINT, {"input": list(texts[i:i + batch_size])})
        out.extend(d["embedding"] for d in sorted(resp["data"], key=lambda d: d["index"]))
    X = np.asarray(out, dtype="float32")
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def chat(messages, model, max_tokens=1000, temperature=0.0, timeout=240, **extra):
    """Returns dict(text, reasoning, finish_reason, usage, latency_s, model). extra -> request body
    (e.g. reasoning_effort="low", response_format={...})."""
    t0 = time.time()
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, **extra}
    resp = _invoke(model, body, timeout=timeout)
    choice = resp["choices"][0]
    content = choice["message"].get("content")
    if isinstance(content, list):                       # reasoning models: list of typed blocks
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        reasoning = "\n".join(s.get("text", "") for b in content if b.get("type") == "reasoning"
                              for s in (b.get("summary") or []))
    else:
        text, reasoning = content or "", ""
    usage = resp.get("usage", {}) or {}
    return {"text": text.strip(), "reasoning": reasoning, "finish_reason": choice.get("finish_reason"),
            "usage": {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
                      "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")},
            "latency_s": round(time.time() - t0, 2), "model": model}

# Pay-per-token prices, DBU per 1M tokens (input, output) — Azure Databricks pricing page, Sept 2026
DBU_PER_M = {"databricks-meta-llama-3-3-70b-instruct": (7.143, 21.429),
             "databricks-qwen3-next-80b-a3b-instruct": (2.143, 17.143),
             "databricks-gpt-oss-120b": (2.143, 8.571),
             "databricks-qwen35-122b-a10b": (3.143, 31.429),
             "databricks-llama-4-maverick": (7.143, 21.429),
             "databricks-gte-large-en": (1.857, 0.0)}
USD_PER_DBU = 0.07                                      # approximate; depends on contract and region

def dbu_cost(model, prompt_tokens, completion_tokens):
    i, o = DBU_PER_M[model]
    return (prompt_tokens * i + completion_tokens * o) / 1e6

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadata filters (same syntax for both backends)
# MAGIC
# MAGIC **What:** filters are a dict, in the format Vector Search Standard endpoints accept:
# MAGIC - `{"country": "Kenya"}` equals, `{"region": ["South Asia", "East Asia and Pacific"]}` is one of,
# MAGIC - `{"approval_fy >=": 2020}`, `{"approval_fy <": 2016}` comparisons, `{"doc_type NOT": "Project Information Document"}` not equal.
# MAGIC
# MAGIC `_apply_filters` implements the same rules on a pandas table for the FAISS backend.
# MAGIC
# MAGIC **Why:** notebooks 10–12 can then pass one filter dict regardless of backend (e.g. "only this `project_id`" when writing a note about one project).

# COMMAND ----------

_OPS = {">=": np.greater_equal, "<=": np.less_equal, ">": np.greater, "<": np.less}

def _apply_filters(df, filters):
    mask = np.ones(len(df), dtype=bool)
    for key, val in (filters or {}).items():
        parts = key.split()
        col, op = parts[0], (parts[1] if len(parts) > 1 else "=")
        s = df[col]
        if op in _OPS:
            m = _OPS[op](s.to_numpy(), val)
        elif op == "NOT":
            m = ~s.isin(val if isinstance(val, (list, tuple)) else [val]).to_numpy()
        elif op == "=":
            m = s.isin(val if isinstance(val, (list, tuple)) else [val]).to_numpy()
        else:
            raise ValueError(f"unsupported filter operator: {key}")
        mask &= m
    return mask

# COMMAND ----------

# MAGIC %md
# MAGIC ## `search()` — one function, two backends
# MAGIC
# MAGIC **What:** embeds the question, then
# MAGIC - **vector_search**: calls the index with `query_type="HYBRID"` (vector + keyword) or `"ANN"` (vector only; ANN = approximate nearest neighbour). For hybrid search with self-computed embeddings, Vector Search needs both the query vector and the query text.
# MAGIC - **faiss**: loads the FAISS index and chunk metadata once (cached), applies filters as an allow-list of row ids, and returns the nearest chunks by cosine similarity. `mode="hybrid"` silently becomes vector-only and the returned `mode` column says so.
# MAGIC
# MAGIC Returns a DataFrame with `rank`, `score`, `cosine`, `backend`, `mode` and the metadata columns. `cosine` is the exact cosine similarity between the question and each returned chunk (for Vector Search computed from the stored vector that the index returns), so a single "is the evidence strong enough?" threshold works for every backend and mode.
# MAGIC
# MAGIC **Why:** the assistant (10), evaluation (11) and notes (12) call `search()` and never touch backend details, so switching to the free FAISS backend is one setting. **Scores are not comparable across backends or modes:** FAISS returns cosine similarity (−1…1), Vector Search ANN returns a similarity derived from distance, and hybrid returns a rank-fusion score (small numbers like 0.01–0.03). Only compare scores within one backend+mode.

# COMMAND ----------

_cache = {}

def _vs_index():
    if "vs" not in _cache:
        from databricks.vector_search.client import VectorSearchClient
        _cache["vs"] = VectorSearchClient(disable_notice=True).get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
    return _cache["vs"]

def _faiss_load():
    if "faiss" not in _cache:
        import faiss, shutil, os
        local = "/local_disk0/tmp/rag_faiss"
        os.makedirs(local, exist_ok=True)
        shutil.copy(f"{FAISS_DIR}/index.faiss", f"{local}/index.faiss")
        index = faiss.read_index(f"{local}/index.faiss")
        ids = json.load(open(f"{FAISS_DIR}/chunk_ids.json"))
        meta = (spark.table(CHUNKS_TABLE).select(*RETURN_COLUMNS).toPandas()
                .set_index("chunk_id").loc[ids].reset_index())
        assert index.ntotal == len(meta), (index.ntotal, len(meta))
        _cache["faiss"] = (index, meta)
    return _cache["faiss"]

def search(query, k=5, filters=None, mode="hybrid", backend=None, columns=None):
    backend = backend or VECTOR_BACKEND
    columns = columns or RETURN_COLUMNS
    qv = embed_texts([query])[0]
    if backend == "vector_search":
        kw = dict(query_vector=qv.tolist(), query_text=query if mode == "hybrid" else None,
                  query_type="HYBRID" if mode == "hybrid" else "ANN", filters=filters, num_results=k, disable_notice=True)
        try:
            res = _vs_index().similarity_search(columns=list(columns) + ["embedding"], **kw)
        except Exception as e:                  # if the vector column cannot be returned, search without it
            print("  (vector column not returned:", str(e)[:120], ")")
            res = _vs_index().similarity_search(columns=list(columns), **kw)
        names = [c["name"] for c in res["manifest"]["columns"]]
        df = pd.DataFrame(res.get("result", {}).get("data_array", []) or [], columns=names)
        used_mode = mode
        if "embedding" in df.columns:          # exact cosine for the abstain rule (hybrid scores are rank-based)
            df["cosine"] = [float(np.dot(np.asarray(e, dtype="float32"), qv)) for e in df.pop("embedding")]
        else:
            df["cosine"] = np.nan
    elif backend == "faiss":
        import faiss
        index, meta = _faiss_load()
        q = qv.reshape(1, -1) / np.linalg.norm(qv)
        params = None
        if filters:
            allowed = np.flatnonzero(_apply_filters(meta, filters)).astype("int64")
            if len(allowed) == 0:
                return pd.DataFrame(columns=list(columns) + ["score", "cosine", "rank", "backend", "mode"])
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))
        scores, rows = index.search(q.astype("float32"), k, params=params)
        keep = rows[0] >= 0
        df = meta.iloc[rows[0][keep]][columns].reset_index(drop=True)
        df["score"] = scores[0][keep]
        df["cosine"] = df["score"]             # FAISS inner product on unit vectors = cosine
        used_mode = "vector"   # FAISS has no keyword index: hybrid is not available
    else:
        raise ValueError(backend)
    df["rank"] = np.arange(1, len(df) + 1)
    df["backend"], df["mode"] = backend, used_mode
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Automatic fallback to FAISS when Vector Search is not available
# MAGIC
# MAGIC **What:** when `VECTOR_BACKEND = "vector_search"`, checks once that the endpoint `wb-rag` exists and is ONLINE and that the index `gold.rag_chunks_index` exists and is ready. If any of that fails (the endpoint was deleted to save money, the client library is missing, no permission ...), it prints a notice and switches `VECTOR_BACKEND` to `"faiss"`, which reads the index file from the volume `gold.rag_index`. **Hybrid (keyword + vector) search is then not available**; every search is vector-only and the returned `mode` column says `vector`. The builder notebook `09` sets `RAG_BUILDER = True` before loading this notebook, which turns the fallback off (09 must create the endpoint, not avoid it).
# MAGIC
# MAGIC **Why:** the Vector Search endpoint bills ~4 DBU/hour while it holds an index, so it may be deleted between demos. With the fallback, reviewers can still run notebooks 10–12 end to end, at no endpoint cost, using the same stored vectors (FAISS and Vector Search ANN returned identical top-5 results in `09`). Recreating the endpoint and index (run `09` with the `vector_search` backend) takes about 30–35 minutes and needs no new embeddings.

# COMMAND ----------

def vector_search_status():
    """(ok, reason): is the Vector Search endpoint ONLINE and the index ready?"""
    try:
        from databricks.vector_search.client import VectorSearchClient
        vsc = VectorSearchClient(disable_notice=True)
        ep = vsc.get_endpoint(VS_ENDPOINT)
        state = (ep.get("endpoint_status") or {}).get("state")
        if state != "ONLINE":
            return False, f"endpoint {VS_ENDPOINT} is {state}"
        st = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX).describe().get("status", {})
        if not st.get("ready"):
            return False, f"index {VS_INDEX} is not ready ({st.get('detailed_state') or st.get('message', '')[:80]})"
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:160]}"

if VECTOR_BACKEND == "vector_search" and not globals().get("RAG_BUILDER", False):
    _ok, _why = vector_search_status()
    if not _ok:
        print(f"NOTICE: Vector Search is not available ({_why}).\n"
              f"        Falling back to the FAISS index in {FAISS_DIR}: searches are vector-only, hybrid search is unavailable.\n"
              f"        To restore Vector Search, run 09_rag_index with vector_backend = vector_search (~30-35 min, no new embeddings).")
        VECTOR_BACKEND = "faiss"

print(f"rag_common loaded: backend={VECTOR_BACKEND}, chunks={CHUNKS_TABLE}, index={VS_INDEX}, endpoint={VS_ENDPOINT}")
