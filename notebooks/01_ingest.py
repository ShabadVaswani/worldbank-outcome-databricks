# Databricks notebook source
# MAGIC %md
# MAGIC # 01_ingest — sources → bronze
# MAGIC
# MAGIC Downloads each source, saves the untouched response to the `bronze.raw_docs` volume (a timestamped snapshot), then loads it into a bronze Delta table.
# MAGIC Nothing is cleaned here — bronze is the raw record. Cleaning and leakage removal happen in `02_structured_silver`.
# MAGIC
# MAGIC | Part | Source | Bronze table |
# MAGIC |---|---|---|
# MAGIC | 1 | IEG project ratings (labels) | `bronze.ieg_ratings` |
# MAGIC | 2 | World Bank Projects API v3 (project features) | `bronze.projects` |
# MAGIC | 3 | World Bank Indicators API: WDI macro + WGI governance (country context) | `bronze.indicators` |
# MAGIC
# MAGIC Documents (PADs etc.) are ingested separately in `01b_ingest_docs`.
# MAGIC
# MAGIC **Attach to:** `wb-ml-single`. Run top to bottom.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings
# MAGIC
# MAGIC **What:** imports and every configurable value in one place: catalog, raw-file folder, API addresses, page sizes.
# MAGIC
# MAGIC **Why:** change a value once here instead of hunting through code; later the job definition can pass these in as parameters.

# COMMAND ----------

import requests, json, time, os
from datetime import datetime, timezone
from pyspark.sql import functions as F

CATALOG = "dbw_wb_outcome"
RAW_ROOT = f"/Volumes/{CATALOG}/bronze/raw_docs"
RUN_STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1 — IEG ratings (the labels)
# MAGIC
# MAGIC **IEG** = Independent Evaluation Group, the World Bank's independent evaluation unit. After a project closes, IEG validates the team's self-assessment (ICR Review, ICRR) or does a field evaluation (PPAR, which supersedes the ICRR) and gives a 6-point **outcome** rating. We'll turn Moderately Satisfactory or better into label 1.
# MAGIC
# MAGIC Source: Finances One dataset `DS00053` — about 12.6k projects, one row per project (latest evaluation only), refreshed quarterly. The API returns at most 5,000 rows per call, so we page.
# MAGIC
# MAGIC ⚠️ **Leakage:** several columns are only known after the project ends (`bank_performance`, `quality_at_entry`, `quality_of_supervision`, `me_quality`, `evaluation_*`, `final_closing_fy`, and `country_fcs_*`, which use closing-year status). They stay in bronze and are dropped in silver.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1a. Download every page
# MAGIC
# MAGIC **What:** requests rows 0–4,999, 5,000–9,999, … until a page comes back short (that's the last one), then checks the total equals the API's own `count`.
# MAGIC
# MAGIC **Why:** paging because of the 5,000-row cap; `raise_for_status()` and the `assert` make a failed or partial download stop loudly instead of silently saving half the data; `timeout` stops a hung request; `sleep(1)` is polite to the API.

# COMMAND ----------

IEG_URL = "https://datacatalogapi.worldbank.org/dexapps/fone/api/apiservice"
IEG_PARAMS = {"datasetId": "DS00053", "resourceId": "RS00049", "type": "json"}
IEG_PAGE = 5000

def fetch_ieg():
    rows, skip = [], 0
    while True:
        r = requests.get(IEG_URL, params={**IEG_PARAMS, "top": IEG_PAGE, "skip": skip}, timeout=60)
        r.raise_for_status()
        body = r.json()
        batch = body["data"]
        rows.extend(batch)
        print(f"skip={skip}: got {len(batch)}  (total {len(rows)} / {body['count']})")
        if len(batch) < IEG_PAGE:
            break
        skip += IEG_PAGE
        time.sleep(1)
    assert len(rows) == body["count"], "row count mismatch - incomplete download"
    return rows

ieg_rows = fetch_ieg()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1b. Save the raw response as a timestamped snapshot
# MAGIC
# MAGIC **What:** writes all rows to `raw_docs/ieg/ieg_ratings_<UTC timestamp>.json`.
# MAGIC
# MAGIC **Why:** this is the untouched original. IEG updates quarterly, so snapshots let you say exactly which version a model was trained on and rebuild without calling the API again.

# COMMAND ----------

ieg_dir = f"{RAW_ROOT}/ieg"
os.makedirs(ieg_dir, exist_ok=True)
ieg_raw_path = f"{ieg_dir}/ieg_ratings_{RUN_STAMP}.json"
with open(ieg_raw_path, "w") as f:
    json.dump(ieg_rows, f)
print("saved", ieg_raw_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1c. Load into the bronze Delta table
# MAGIC
# MAGIC **What:** Spark reads the JSON file, adds `_ingested_at` (load time) and `_source_file`, and saves `bronze.ieg_ratings`.
# MAGIC
# MAGIC **Why:** a Delta table is queryable with SQL, versioned, and tracked by Unity Catalog lineage. `multiLine` because the file is one big JSON array. `overwrite` makes re-runs replace rather than duplicate (idempotent); `overwriteSchema` tolerates IEG adding a column.

# COMMAND ----------

ieg_df = (spark.read.option("multiLine", True).json(ieg_raw_path)
          .withColumn("_ingested_at", F.current_timestamp())
          .withColumn("_source_file", F.lit(ieg_raw_path)))

(ieg_df.write.mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable(f"{CATALOG}.bronze.ieg_ratings"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1d. Sanity checks
# MAGIC
# MAGIC **What:** row count, how many projects per outcome rating, and any project ID that appears more than once.
# MAGIC
# MAGIC **Why:** never assume a write worked. These catch missing rows, unexpected label values, and duplicate keys (which would break joins). Expected: ~12.6k rows, Satisfactory largest, ~150 Not Rated, one duplicate ID.

# COMMAND ----------

ieg_t = spark.table(f"{CATALOG}.bronze.ieg_ratings")
print("rows:", ieg_t.count())
display(ieg_t.groupBy("outcome").count().orderBy(F.desc("count")))
display(ieg_t.groupBy("project_id").count().filter("count > 1"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2 — Projects API (project features)
# MAGIC
# MAGIC Source: `search.worldbank.org/api/v3/projects` — about 28k projects, at most 1,000 per call. (v2 stopped updating in Dec 2024, so we use v3.) Every IEG-rated project has a match here.
# MAGIC
# MAGIC ⚠️ **Leakage:** `closingdate` is the *current* closing date (extensions signal trouble), `status` reflects today, and the effectiveness / ICR milestone dates come after approval. `pdo` (development objective) can be rewritten during restructuring. All kept in bronze, handled in silver.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a. Download every page, saving each page as it arrives
# MAGIC
# MAGIC **What:** requests projects 0–999, 1,000–1,999, … up to the API's `total`, writing each page to its own file in a timestamped folder `raw_docs/projects/<UTC timestamp>/`. Then checks the count.
# MAGIC
# MAGIC **Why:** `fl` asks only for the fields we may use (~53 MB instead of ~270 MB). The API returns projects as a dict keyed by ID, so we convert to a list of records for Spark. Page-by-page saving keeps memory low and keeps finished pages if a later request fails. Takes ~2–4 minutes.

# COMMAND ----------

PROJ_URL = "https://search.worldbank.org/api/v3/projects"
PROJ_PAGE = 1000
PROJ_FIELDS = [
    "id", "project_name", "countryshortname", "countrycode", "regionname",
    "boardapprovaldate", "fiscalyear", "status", "prodline_exact",
    "totalamt", "grantamt", "idacommamt", "lendprojectcost",
    "major_sector_name", "sector_name", "sector_percent", "themecode",
    "borrower", "impagency", "envassesmentcategorycode", "milestones",
    "pdo", "project_abstract", "parentprojid", "closingdate",
]
proj_dir = f"{RAW_ROOT}/projects/{RUN_STAMP}"

def fetch_projects(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    offset, total, n = 0, None, 0
    while total is None or offset < total:
        r = requests.get(PROJ_URL, timeout=120, params={
            "format": "json", "rows": PROJ_PAGE, "os": offset,
            "fl": ",".join(PROJ_FIELDS)})
        r.raise_for_status()
        body = r.json()
        total = int(body["total"])
        page = list(body["projects"].values())
        if not page:
            break
        with open(f"{out_dir}/page_{offset:06d}.json", "w") as f:
            json.dump(page, f)
        n += len(page)
        print(f"os={offset}: {len(page)}  (so far {n} / {total})")
        offset += PROJ_PAGE
        time.sleep(1)
    assert n == total, f"got {n} projects, expected {total}"
    return n

fetch_projects(proj_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. Load all pages into the bronze Delta table
# MAGIC
# MAGIC **What:** Spark reads every page file at once (`*.json`), infers columns, adds `_ingested_at` and `_source_file` (from Spark's hidden `_metadata.file_path`), and saves `bronze.projects`.
# MAGIC
# MAGIC **Why:** one table for all projects; `_source_file` lets you trace any odd row back to its raw page. No cleaning yet — amounts stay as text and `milestones` stays nested.

# COMMAND ----------

proj_df = (spark.read.option("multiLine", True).json(f"{proj_dir}/*.json")
           .withColumn("_ingested_at", F.current_timestamp())
           .withColumn("_source_file", F.col("_metadata.file_path")))

(proj_df.write.mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable(f"{CATALOG}.bronze.projects"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2c. Sanity checks — does the join key work?
# MAGIC
# MAGIC **What:** row and distinct-ID counts, then how many rated IEG projects have a matching project row (a `left_semi` join keeps IEG rows that match, without adding columns), plus 5 sample rows.
# MAGIC
# MAGIC **Why:** proves `project_id` ↔ `id` joins cleanly *now*, instead of discovering a format mismatch in gold. Expected: ~28.2k rows, distinct = rows, matched ≈ all rated IEG projects (~12.5k).

# COMMAND ----------

proj_t = spark.table(f"{CATALOG}.bronze.projects")
print("rows:", proj_t.count(), "| distinct ids:", proj_t.select("id").distinct().count())

ieg_rated = (spark.table(f"{CATALOG}.bronze.ieg_ratings")
             .filter("outcome != 'Not Rated'").select("project_id").distinct())
matched = ieg_rated.join(proj_t, ieg_rated.project_id == proj_t.id, "left_semi").count()
print(f"IEG-rated projects found in projects table: {matched} / {ieg_rated.count()}")

display(proj_t.select("id", "boardapprovaldate", "totalamt", "major_sector_name", "pdo").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3 — Country indicators (macro + governance context)
# MAGIC
# MAGIC Source: World Bank Indicators API `api.worldbank.org/v2` (the same API the `wbgapi` package wraps — calling it directly means no extra install in scheduled jobs).
# MAGIC
# MAGIC | Code | Meaning | Source |
# MAGIC |---|---|---|
# MAGIC | `NY.GDP.PCAP.KD` | GDP per capita, constant 2015 US$ | WDI (2) |
# MAGIC | `NY.GDP.MKTP.KD.ZG` | GDP growth, annual % | WDI (2) |
# MAGIC | `FP.CPI.TOTL.ZG` | Inflation, consumer prices % | WDI (2) |
# MAGIC | `DT.DOD.DECT.GN.ZS` | External debt stocks, % of GNI | WDI (2) |
# MAGIC | `SH.XPD.CHEX.GD.ZS` | Health expenditure, % of GDP | WDI (2) |
# MAGIC | `SE.XPD.TOTL.GD.ZS` | Government education spending, % of GDP | WDI (2) |
# MAGIC | `SP.POP.TOTL` | Population | WDI (2) |
# MAGIC | `NE.TRD.GNFS.ZS` | Trade, % of GDP | WDI (2) |
# MAGIC | `BX.KLT.DINV.WD.GD.ZS` | FDI net inflows, % of GDP | WDI (2) |
# MAGIC | `GOV_WGI_{GE,RQ,RL,CC,PV,VA}.EST` | Worldwide Governance Indicators (≈ −2.5 … +2.5): government effectiveness, regulatory quality, rule of law, control of corruption, political stability, voice & accountability | WGI (3) |
# MAGIC
# MAGIC Note: the old WGI codes (`GE.EST`, …) were archived; the current ones are `GOV_WGI_*`. WGI starts in 1996 and was every two years until 2002.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Download each indicator for all countries, 1980–2025
# MAGIC
# MAGIC **What:** for each code, pages through `country/all/indicator/<code>` (5,000 rows per page) and saves the raw JSON per indicator under `raw_docs/indicators/<UTC timestamp>/`.
# MAGIC
# MAGIC **Why:** one file per indicator keeps the raw record traceable; the API rejects very large pages, so 5,000 is a safe size. We fetch from 1980 so that "the year before approval" exists even for early projects.

# COMMAND ----------

IND_URL = "https://api.worldbank.org/v2/country/all/indicator/{code}"
WDI_CODES = ["NY.GDP.PCAP.KD", "NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "DT.DOD.DECT.GN.ZS",
             "SH.XPD.CHEX.GD.ZS", "SE.XPD.TOTL.GD.ZS", "SP.POP.TOTL", "NE.TRD.GNFS.ZS",
             "BX.KLT.DINV.WD.GD.ZS"]
WGI_CODES = [f"GOV_WGI_{x}.EST" for x in ["GE", "RQ", "RL", "CC", "PV", "VA"]]
IND_SOURCES = {**{c: 2 for c in WDI_CODES}, **{c: 3 for c in WGI_CODES}}
ind_dir = f"{RAW_ROOT}/indicators/{RUN_STAMP}"

def fetch_indicator(code, source):
    rows, page = [], 1
    while True:
        r = requests.get(IND_URL.format(code=code), timeout=120, params={
            "format": "json", "per_page": 5000, "page": page, "date": "1980:2025", "source": source})
        r.raise_for_status()
        meta, data = r.json()
        rows.extend(data or [])
        if page >= int(meta["pages"]):
            break
        page += 1
        time.sleep(0.5)
    assert len(rows) == int(meta["total"]), f"{code}: got {len(rows)}, expected {meta['total']}"
    return rows

os.makedirs(ind_dir, exist_ok=True)
for code, source in IND_SOURCES.items():
    rows = fetch_indicator(code, source)
    with open(f"{ind_dir}/{code}.json", "w") as f:
        json.dump(rows, f)
    print(f"{code}: {len(rows)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. Load into one long bronze table
# MAGIC
# MAGIC **What:** reads all indicator files and keeps one row per (indicator, country, year, value), flattening the nested `indicator` / `country` objects into plain columns.
# MAGIC
# MAGIC **Why:** "long" format (one value per row) is how the API delivers it and is easy to append to; the pivot to one-row-per-country-year happens in silver. Flattening is not cleaning — every value is kept as delivered, nulls included.

# COMMAND ----------

ind_df = (spark.read.option("multiLine", True).json(f"{ind_dir}/*.json")
          .select(F.col("indicator.id").alias("indicator_id"),
                  F.col("indicator.value").alias("indicator_name"),
                  F.col("country.id").alias("country_iso2"),
                  F.col("countryiso3code").alias("country_iso3"),
                  F.col("country.value").alias("country_name"),
                  F.col("date").alias("year"),
                  F.col("value").cast("double").alias("value"))
          .withColumn("_ingested_at", F.current_timestamp())
          .withColumn("_source_file", F.col("_metadata.file_path")))

(ind_df.write.mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable(f"{CATALOG}.bronze.indicators"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3c. Sanity checks
# MAGIC
# MAGIC **What:** per indicator: rows, non-null values, and first/last year with data.
# MAGIC
# MAGIC **Why:** shows coverage at a glance — e.g. WGI should start in 1996 and health spending around 2000 — so we know which features will be missing for older projects.

# COMMAND ----------

ind_t = spark.table(f"{CATALOG}.bronze.indicators")
ind_summary = (ind_t.groupBy("indicator_id")
               .agg(F.count("*").alias("rows"), F.count("value").alias("non_null"),
                    F.min(F.when(F.col("value").isNotNull(), F.col("year"))).alias("first_year"),
                    F.max(F.when(F.col("value").isNotNull(), F.col("year"))).alias("last_year"))
               .orderBy("indicator_id"))
display(ind_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run summary
# MAGIC
# MAGIC **What:** returns the key counts as the notebook's exit value.
# MAGIC
# MAGIC **Why:** when this notebook runs as a job task, the exit value shows up in the run output, so a scheduled run can be checked without opening the notebook.

# COMMAND ----------

summary = {
    "ieg_rows": ieg_t.count(),
    "projects_rows": proj_t.count(),
    "ieg_rated_matched": matched,
    "indicator_rows": ind_t.count(),
    "indicators": ind_t.select("indicator_id").distinct().count(),
}
print(summary)
dbutils.notebook.exit(json.dumps(summary))
