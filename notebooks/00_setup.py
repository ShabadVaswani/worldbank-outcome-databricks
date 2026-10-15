# Databricks notebook source
# MAGIC %md
# MAGIC # 00_setup — one-time workspace setup
# MAGIC
# MAGIC Creates the Unity Catalog schemas and volume the project uses, then smoke-tests the libraries and the World Bank API.
# MAGIC Safe to re-run: every step is idempotent.
# MAGIC
# MAGIC **Attach to:** `wb-ml-single` (18 LTS ML, single node). Not Serverless.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Schemas and volume (medallion layout)
# MAGIC
# MAGIC **What:** creates `bronze` (raw data as ingested), `silver` (cleaned, leakage-safe), `gold` (model-ready features), `ml` (registered models), and the `bronze.raw_docs` volume for raw files (API JSON snapshots, PDFs).
# MAGIC
# MAGIC **Why:** separating layers means a bug in silver is fixed by re-running silver from bronze — no API calls needed. `IF NOT EXISTS` makes re-runs harmless; `COMMENT`s document the catalog for anyone browsing it.

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG dbw_wb_outcome;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS bronze COMMENT 'Raw data as ingested from World Bank APIs and documents';
# MAGIC CREATE SCHEMA IF NOT EXISTS silver COMMENT 'Cleaned, standardized, leakage-safe data';
# MAGIC CREATE SCHEMA IF NOT EXISTS gold   COMMENT 'Model-ready feature tables keyed by project_id';
# MAGIC CREATE SCHEMA IF NOT EXISTS ml     COMMENT 'Registered ML models';
# MAGIC
# MAGIC CREATE VOLUME IF NOT EXISTS bronze.raw_docs COMMENT 'Raw PAD/PID PDFs and source files';

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Install the three libraries the ML runtime doesn't include
# MAGIC
# MAGIC **What:** installs `wbgapi` (World Bank Indicators API client), `pymupdf` (PDF text extraction), `faiss-cpu` (vector search for the optional RAG step), then restarts Python so they can be imported.
# MAGIC
# MAGIC **Why:** 18 LTS ML already ships xgboost, MLflow, optuna, shap, and sentence-transformers. These installs are notebook-scoped and vanish when the cluster restarts — later the job definition will install them automatically.

# COMMAND ----------

# MAGIC %pip install wbgapi pymupdf faiss-cpu

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Smoke test
# MAGIC
# MAGIC **What:** imports every library the pipeline needs, calls the World Bank API for Kenya, and writes/reads/deletes a test file in the volume.
# MAGIC
# MAGIC **Why:** proves the runtime, internet access, and volume permissions all work *before* writing real pipeline code. Expect xgboost 3.1.x and mlflow 3.8.x. The CUDA/TensorFlow log lines are harmless (no GPU on this cluster).

# COMMAND ----------

import xgboost, mlflow, optuna, shap, sentence_transformers
import wbgapi as wb, pymupdf, faiss

print("xgboost", xgboost.__version__, "| mlflow", mlflow.__version__)
print("sentence-transformers", sentence_transformers.__version__)

print(wb.economy.get("KEN"))

path = "/Volumes/dbw_wb_outcome/bronze/raw_docs/_smoke_test.txt"
with open(path, "w") as f:
    f.write("hello volume")
print(open(path).read())
dbutils.fs.rm(path)
