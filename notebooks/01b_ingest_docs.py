# Databricks notebook source
# MAGIC %md
# MAGIC # 01b_ingest_docs — pre-approval documents → bronze
# MAGIC
# MAGIC The unstructured half of the project. For each project we want **one document written before the Board approved it**, so the text describes the plan, never the result.
# MAGIC
# MAGIC | Step | Output |
# MAGIC |---|---|
# MAGIC | 1. Document catalog (metadata for 6 pre-approval document types) | `bronze.doc_metadata` |
# MAGIC | 2. Pick one document per project (leakage-safe rules) | `bronze.doc_manifest` |
# MAGIC | 3. Download the plain-text version of each picked document | files in `raw_docs/docs/`, log in `bronze.doc_downloads` |
# MAGIC
# MAGIC Source: World Bank Documents & Reports API `search.worldbank.org/api/v2/wds`. Every document has a `.txt` version (the Bank's own text extraction), so we don't need to parse PDFs.
# MAGIC
# MAGIC **Needs:** `01_ingest` run first (uses `bronze.ieg_ratings` and `bronze.projects`). **Attach to:** `wb-ml-single`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings
# MAGIC
# MAGIC **What:** catalog/paths, the document types we accept (in priority order), and the scope of projects.
# MAGIC
# MAGIC **Why:**
# MAGIC - **Allowed types are all pre-approval:** PAD (modern appraisal document), Staff Appraisal Report (the pre-~2000 name for a PAD), Program Document (for budget-support loans), PID (short project summary), and the President's memorandum/report to the Board. Completion reports (ICR) and reviews (ICRR) are **never** allowed — they contain the outcome.
# MAGIC - **Scope:** rated projects approved FY1995 or later (governance indicators start in 1996; older text is often poor OCR) plus active projects approved FY2019+ (for batch scoring in `07_deploy`).

# COMMAND ----------

import requests, json, time, os
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyspark.sql import functions as F, Window

CATALOG = "dbw_wb_outcome"
RAW_ROOT = f"/Volumes/{CATALOG}/bronze/raw_docs"
RUN_STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

WDS_URL = "https://search.worldbank.org/api/v2/wds"
DOC_TYPES = [  # priority order: first = best
    "Project Appraisal Document",
    "Staff Appraisal Report",
    "Program Document",
    "Project Information Document",
    "Memorandum & Recommendation of the President",
    "President's Report",
]
MIN_RATED_FY = 1995
MIN_ACTIVE_FY = 2019
DOCS_DIR = f"{RAW_ROOT}/docs"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Document catalog
# MAGIC
# MAGIC **What:** for each allowed type, pages through the documents API (1,000 per call) and saves the raw metadata pages, then loads them into `bronze.doc_metadata`.
# MAGIC
# MAGIC **Why:** pulling the catalog by *type* (≈36k documents in ~40 calls) is far cheaper than one call per project (~14k calls). `fl` limits the fields to what we need: project ID, document date, type, language, and the text URL.

# COMMAND ----------

meta_dir = f"{RAW_ROOT}/doc_metadata/{RUN_STAMP}"
os.makedirs(meta_dir, exist_ok=True)

def fetch_doc_type(doc_type, idx):
    offset, n, total = 0, 0, None
    while total is None or offset < total:
        r = requests.get(WDS_URL, timeout=120, params={
            "format": "json", "rows": 1000, "os": offset, "docty_exact": doc_type,
            "fl": "projectid,docdt,docty,lang,txturl,pdfurl,display_title"})
        r.raise_for_status()
        body = r.json()
        total = int(body["total"])
        page = [v for k, v in body["documents"].items() if k != "facets"]
        if not page:
            break
        with open(f"{meta_dir}/type{idx}_{offset:06d}.json", "w") as f:
            json.dump(page, f)
        n += len(page)
        offset += 1000
        time.sleep(0.5)
    print(f"{doc_type}: {n} / {total}")
    return n

for i, t in enumerate(DOC_TYPES):
    fetch_doc_type(t, i)

meta_df = (spark.read.option("multiLine", True).json(f"{meta_dir}/*.json")
           .withColumn("_ingested_at", F.current_timestamp())
           .withColumn("_source_file", F.col("_metadata.file_path")))
(meta_df.write.mode("overwrite").option("overwriteSchema", "true")
   .saveAsTable(f"{CATALOG}.bronze.doc_metadata"))
print("doc_metadata rows:", spark.table(f"{CATALOG}.bronze.doc_metadata").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Pick one document per project (the leakage guard)
# MAGIC
# MAGIC **What:** joins documents to projects and keeps, per project, the best document that is
# MAGIC 1. English, with a text URL,
# MAGIC 2. dated **on or before the Board approval date**,
# MAGIC 3. the highest-priority type available; ties go to the latest such document (closest to approval, so most complete).
# MAGIC
# MAGIC **Why:** rule 2 is the core leakage protection for the text features — a document dated after approval could describe implementation problems. A document can list several project IDs (`P1,P2`), so we split them first.

# COMMAND ----------

prio = F.create_map(*[x for i, t in enumerate(DOC_TYPES) for x in (F.lit(t), F.lit(i))])

docs = (spark.table(f"{CATALOG}.bronze.doc_metadata")
        .withColumn("project_id", F.explode(F.split(F.col("projectid"), ",")))
        .withColumn("project_id", F.trim("project_id"))
        .withColumn("doc_date", F.to_date(F.substring("docdt", 1, 10)))
        .withColumn("priority", prio[F.col("docty")])
        .filter(F.col("txturl").isNotNull() & F.col("lang").contains("English")))

proj = (spark.table(f"{CATALOG}.bronze.projects")
        .select(F.col("id").alias("project_id"),
                F.to_date(F.substring("boardapprovaldate", 1, 10)).alias("approval_date"),
                F.col("fiscalyear").cast("int").alias("approval_fy"),
                F.col("status")))

rated = (spark.table(f"{CATALOG}.bronze.ieg_ratings")
         .filter("outcome != 'Not Rated'").select("project_id").distinct()
         .withColumn("is_rated", F.lit(True)))

in_scope = (proj.join(rated, "project_id", "left")
            .filter((F.col("is_rated") & (F.col("approval_fy") >= MIN_RATED_FY)) |
                    ((F.col("status") == "Active") & (F.col("approval_fy") >= MIN_ACTIVE_FY)))
            .withColumn("is_rated", F.coalesce("is_rated", F.lit(False))))

w = Window.partitionBy("project_id").orderBy(F.col("priority").asc(), F.col("doc_date").desc(), F.col("id").asc())
manifest = (in_scope.join(docs, "project_id", "inner")
            .filter(F.col("doc_date") <= F.col("approval_date"))
            .withColumn("rn", F.row_number().over(w)).filter("rn = 1")
            .select("project_id", "is_rated", "approval_date", "approval_fy",
                    F.col("id").alias("doc_id"), F.col("docty").alias("doc_type"), "doc_date",
                    F.regexp_replace("txturl", "^http://", "https://").alias("txt_url")))

(manifest.write.mode("overwrite").option("overwriteSchema", "true")
   .saveAsTable(f"{CATALOG}.bronze.doc_manifest"))

m = spark.table(f"{CATALOG}.bronze.doc_manifest")
print("projects in scope:", in_scope.count(), "| with a pre-approval document:", m.count())
display(m.groupBy("is_rated", "doc_type").count().orderBy("is_rated", F.desc("count")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Download the text files (parallel, resumable)
# MAGIC
# MAGIC **What:** downloads each picked document's `.txt` to `raw_docs/docs/<project_id>__<doc_id>.txt` using 8 parallel threads, skipping files that already exist, and logs success/failure per document to `bronze.doc_downloads`.
# MAGIC
# MAGIC **Why:**
# MAGIC - **Threads:** each download is ~0.5 s of waiting on the network, so 8 at a time cuts ~1 hour to ~10–15 minutes without hammering the server.
# MAGIC - **Skip-if-exists:** re-running only fetches what's missing — the download is idempotent and survives interruptions.
# MAGIC - **Log table:** failed downloads are recorded instead of crashing the run; the text pipeline simply has no text for those projects.

# COMMAND ----------

os.makedirs(DOCS_DIR, exist_ok=True)
todo = m.select("project_id", "doc_id", "txt_url").collect()
session = requests.Session()

def download(row):
    path = f"{DOCS_DIR}/{row.project_id}__{row.doc_id}.txt"
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return (row.project_id, row.doc_id, path, "exists", os.path.getsize(path))
    for attempt in range(3):
        try:
            r = session.get(row.txt_url, timeout=90)
            if r.status_code == 200 and r.content:
                with open(path, "wb") as f:
                    f.write(r.content)
                return (row.project_id, row.doc_id, path, "ok", len(r.content))
            status = f"http_{r.status_code}"
        except Exception as e:
            status = f"error_{type(e).__name__}"
        time.sleep(2 * (attempt + 1))
    return (row.project_id, row.doc_id, None, status, 0)

results = []
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(download, r) for r in todo]
    for i, fut in enumerate(as_completed(futures), 1):
        results.append(fut.result())
        if i % 500 == 0:
            print(f"{i}/{len(todo)} done ({time.time() - t0:.0f}s)")

dl = (spark.createDataFrame(results, "project_id string, doc_id string, path string, status string, bytes long")
      .withColumn("_ingested_at", F.current_timestamp()))
(dl.write.mode("overwrite").option("overwriteSchema", "true")
   .saveAsTable(f"{CATALOG}.bronze.doc_downloads"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sanity checks
# MAGIC
# MAGIC **What:** download outcomes by status, file-size distribution, and how many rated projects now have text.
# MAGIC
# MAGIC **Why:** tiny files usually mean a failed or empty extraction; the rated-with-text count is the size of the dataset the text model can learn from.

# COMMAND ----------

d = spark.table(f"{CATALOG}.bronze.doc_downloads")
display(d.groupBy("status").count())
display(d.filter("path is not null").select(
    F.expr("percentile(bytes, array(0.05, 0.5, 0.95))").alias("bytes_p05_p50_p95")))

summary = {
    "doc_metadata_rows": spark.table(f"{CATALOG}.bronze.doc_metadata").count(),
    "manifest_rows": m.count(),
    "manifest_rated": m.filter("is_rated").count(),
    "downloaded_ok": d.filter("status in ('ok','exists')").count(),
    "download_failed": d.filter("status not in ('ok','exists')").count(),
}
print(summary)
dbutils.notebook.exit(json.dumps(summary))
