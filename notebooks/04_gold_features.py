# Databricks notebook source
# MAGIC %md
# MAGIC # 04_gold_features — one model-ready row per project
# MAGIC
# MAGIC Joins everything into `gold.project_features` (primary key `project_id`, so Unity Catalog treats it as a **feature table**):
# MAGIC
# MAGIC ```
# MAGIC silver.projects --+-- silver.labels            (label, if rated)
# MAGIC                   |-- silver.country_year      (year BEFORE approval)  <-- leakage rule
# MAGIC                   |-- silver.doc_text_features (pre-approval doc)      <-- leakage rule
# MAGIC                   +-- silver.doc_embeddings -> PCA(20), fit on TRAIN only
# MAGIC ```
# MAGIC
# MAGIC Each row gets a `split`:
# MAGIC | split | rows | used for |
# MAGIC |---|---|---|
# MAGIC | `train` | rated, approved FY1995–2011 | fitting and tuning |
# MAGIC | `test` | rated, approved FY2012+ | the honest final score — "projects from the future" |
# MAGIC | `score` | active, unrated, approved FY2019+ | batch predictions in `07_deploy` |
# MAGIC
# MAGIC **Attach to:** `wb-ml-single`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings
# MAGIC
# MAGIC **What:** split years, PCA size, and the minimum count for a category to get its own one-hot column.
# MAGIC
# MAGIC **Why:** a **time-based split** mimics real use — you train on the past and predict projects approved later. A random split would let the model learn from projects approved *after* the ones it's tested on, which inflates scores. Rare categories are grouped into `other` so the model doesn't memorise one-off values.

# COMMAND ----------

import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pyspark.sql import functions as F

CATALOG = "dbw_wb_outcome"
MIN_FY, TRAIN_MAX_FY, SCORE_MIN_FY = 1995, 2011, 2019
N_PCA = 20
MIN_CAT_COUNT = 30
CATEGORICALS = ["region", "major_sector", "env_category", "doc_type"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Assemble the base table
# MAGIC
# MAGIC **What:** starts from projects, attaches the label, the country indicators for **approval year − 1**, and the text features; then assigns `split` and drops rows outside the three groups.
# MAGIC
# MAGIC **Why:** joining indicators on `year = approval_year - 1` is the macro leakage guard: a project approved in March 2005 sees 2004 data, which was (mostly) published by then — never 2005 values that describe conditions during implementation.

# COMMAND ----------

proj = spark.table(f"{CATALOG}.silver.projects")
labels = spark.table(f"{CATALOG}.silver.labels").select("project_id", "label", "ieg_outcome")
cy = spark.table(f"{CATALOG}.silver.country_year")
tf = spark.table(f"{CATALOG}.silver.doc_text_features").drop("doc_id")

base = (proj.join(labels, "project_id", "left")
        .withColumn("indicator_year", F.year("approval_date") - 1)
        .join(cy.withColumnRenamed("year", "indicator_year"), ["country_iso2", "indicator_year"], "left")
        .join(tf, "project_id", "left")
        .withColumn("split",
            F.when(F.col("label").isNotNull() & F.col("approval_fy").between(MIN_FY, TRAIN_MAX_FY), "train")
             .when(F.col("label").isNotNull() & (F.col("approval_fy") > TRAIN_MAX_FY), "test")
             .when(F.col("label").isNull() & (F.col("status") == "Active") & (F.col("approval_fy") >= SCORE_MIN_FY), "score"))
        .filter(F.col("split").isNotNull()))

pdf = base.toPandas()
emb = spark.table(f"{CATALOG}.silver.doc_embeddings").toPandas()
print(pdf["split"].value_counts().to_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Engineered structured features
# MAGIC
# MAGIC **What:** log-scales money (amounts span 4 orders of magnitude), flags approvals in June (the last month of the Bank's fiscal year), and adds a `has_text` flag.
# MAGIC
# MAGIC **Why:** June approvals are a known "end-of-year rush" pattern; `has_text` lets the model tell "no document" apart from "a document with low keyword counts". All are known on approval day.

# COMMAND ----------

pdf["log_commitment"] = np.log1p(pdf["commitment_musd"].clip(lower=0))
pdf["log_project_cost"] = np.log1p(pdf["project_cost_musd"].clip(lower=0))
pdf["approved_in_june"] = (pd.to_datetime(pdf["approval_date"]).dt.month == 6).astype(float)
pdf["has_text"] = pdf["n_words"].notna().astype(float)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2b. Track-record features (only ratings published before approval)
# MAGIC
# MAGIC **What:** for each project, looks at **other** projects in the same country (and, separately, the same sector and region) whose IEG rating was **published before this project's approval year**, and computes:
# MAGIC - `*_prior_n` — how many such rated projects existed,
# MAGIC - `*_prior_success` — their success rate, **smoothed** toward the overall success rate known at that time: (successes + 10 × overall_rate) / (n + 10).
# MAGIC
# MAGIC **Why:** World Bank research finds country-level conditions explain a large share of project outcomes; a country's recent track record is the most direct summary of that, and a Bank analyst would know it on approval day. The strict "published before approval" filter is the leakage guard — the project's own rating (published years later) can never be included. Smoothing stops a country with 1 rated project from getting a rate of 0% or 100%.

# COMMAND ----------

SMOOTH = 10
lab_hist = (spark.table(f"{CATALOG}.silver.labels").select("project_id", "label", "rating_published_fy")
            .join(proj.select("project_id", "country_iso2", "major_sector", "region"), "project_id")
            .toPandas().dropna(subset=["rating_published_fy"]))
lab_hist["rating_published_fy"] = lab_hist["rating_published_fy"].astype(int)

# overall success rate known before each year
years = range(1980, 2031)
yearly = lab_hist.groupby("rating_published_fy")["label"].agg(["sum", "count"]).reindex(years, fill_value=0).cumsum()
overall_before = {y: (yearly.loc[y - 1, "sum"] / yearly.loc[y - 1, "count"]) if yearly.loc[y - 1, "count"] else 0.7
                  for y in years if y - 1 in yearly.index}

def track_record(key, prefix):
    g = (lab_hist.dropna(subset=[key]).groupby([key, "rating_published_fy"])["label"].agg(["sum", "count"]))
    out_n, out_rate = [], []
    for k, fy in zip(pdf[key], pdf["approval_fy"]):
        if pd.isna(k) or pd.isna(fy) or k not in g.index.get_level_values(0):
            n, s = 0, 0.0
        else:
            sub = g.loc[k]
            prior = sub[sub.index < int(fy)]
            n, s = int(prior["count"].sum()), float(prior["sum"].sum())
        base = overall_before.get(int(fy), 0.7) if not pd.isna(fy) else 0.7
        out_n.append(float(n))
        out_rate.append((s + SMOOTH * base) / (n + SMOOTH))
    pdf[f"{prefix}_prior_n"] = out_n
    pdf[f"{prefix}_prior_success"] = out_rate

track_record("country_iso2", "country")
track_record("major_sector", "sector")
track_record("region", "region")
display(pdf.groupby("split")[["country_prior_n", "country_prior_success", "sector_prior_success"]].mean().round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Embedding → 20 principal components (fit on train only)
# MAGIC
# MAGIC **What:** fits PCA on the **train** projects' 384-dim embeddings, then projects every project onto the top 20 components (`emb_pc_01` … `emb_pc_20`).
# MAGIC
# MAGIC **Why:** 384 columns for ~5k training rows invites overfitting; 20 components keep most of the variation. Fitting on train only means nothing about the test period shapes the features.

# COMMAND ----------

emb_map = dict(zip(emb["project_id"], emb["embedding"]))
has_emb = pdf["project_id"].isin(emb_map)
X_emb = np.stack([np.asarray(emb_map[p], dtype="float32") for p in pdf.loc[has_emb, "project_id"]])
train_mask = (pdf.loc[has_emb, "split"] == "train").values

pca = PCA(n_components=N_PCA, random_state=42).fit(X_emb[train_mask])
pcs = pca.transform(X_emb)
pc_cols = [f"emb_pc_{i+1:02d}" for i in range(N_PCA)]
pc_df = pd.DataFrame(pcs, columns=pc_cols, index=pdf.index[has_emb])
pdf = pdf.join(pc_df)
print("explained variance (20 PCs):", round(float(pca.explained_variance_ratio_.sum()), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3b. Supervised text score: TF-IDF + logistic regression (out-of-fold)
# MAGIC
# MAGIC **What:** turns each document's embedding-window text (~1,600 words) into TF-IDF word/bigram weights, fits a regularised logistic regression to predict success, and stores its probability as one feature, `text_lr_score`.
# MAGIC - **Train rows** get **time-respecting out-of-fold** scores, using the same year blocks as the rolling CV in `05_train`: projects approved FY2002–04 are scored by a model fitted on FY1995–2001, FY2005–07 by a model fitted on FY1995–2004, FY2008–11 by FY1995–2007. FY1995–2001 have no earlier data, so their score is missing (XGBoost handles that).
# MAGIC - **Test and score rows** are scored by one model fitted on all train rows (vocabulary also learned from train only).
# MAGIC
# MAGIC **Why:** PCA'd embeddings mostly capture *topic* (sector, region) — which structured features already know. A supervised text model instead learns which specific words and phrases go with success or failure. Feeding its score to XGBoost is called **stacking**.
# MAGIC
# MAGIC **Lesson learned (kept on purpose):** a first version used *random* 5-fold out-of-fold scores. That let a 2003 project's score come from a model that had seen 2009 projects, so validation AUC for text models was ~0.05 too optimistic versus test, and the champion choice was biased toward them. Out-of-fold must respect time when validation does.

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

docs_text = (spark.table(f"{CATALOG}.silver.doc_chunks").groupBy("project_id")
             .agg(F.concat_ws(" ", F.sort_array(F.collect_list(F.struct("chunk_idx", "chunk_text"))).getField("chunk_text")).alias("text"))
             .toPandas())
pdf = pdf.merge(docs_text, on="project_id", how="left")
has_txt = pdf["text"].notna()
tr_idx = pdf.index[has_txt & (pdf["split"] == "train")]
other_idx = pdf.index[has_txt & (pdf["split"] != "train")]

def tfidf_lr():
    return (TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), max_features=50000,
                            sublinear_tf=True, stop_words="english"),
            LogisticRegression(C=0.5, max_iter=3000))

OOF_BLOCKS = [(2002, 2004), (2005, 2007), (2008, 2011)]   # must match FOLDS in 05_train
pdf["text_lr_score"] = np.nan
for a, b in OOF_BLOCKS:
    fit_idx = tr_idx[pdf.loc[tr_idx, "approval_fy"].values < a]
    blk_idx = tr_idx[pdf.loc[tr_idx, "approval_fy"].between(a, b).values]
    vec, lr = tfidf_lr()
    lr.fit(vec.fit_transform(pdf.loc[fit_idx, "text"]), pdf.loc[fit_idx, "label"].astype(int))
    pdf.loc[blk_idx, "text_lr_score"] = lr.predict_proba(vec.transform(pdf.loc[blk_idx, "text"]))[:, 1]
vec, lr = tfidf_lr()
lr.fit(vec.fit_transform(pdf.loc[tr_idx, "text"]), pdf.loc[tr_idx, "label"].astype(int))
pdf.loc[other_idx, "text_lr_score"] = lr.predict_proba(vec.transform(pdf.loc[other_idx, "text"]))[:, 1]
pdf = pdf.drop(columns="text")

from sklearn.metrics import roc_auc_score
oof = pdf.loc[tr_idx].dropna(subset=["text_lr_score"])
te = pdf[(pdf["split"] == "test") & pdf["text_lr_score"].notna()]
print(f"text_lr_score AUC  time-OOF (FY2002-11): {roc_auc_score(oof['label'], oof['text_lr_score']):.3f}   "
      f"test: {roc_auc_score(te['label'], te['text_lr_score']):.3f}")
terms = np.array(vec.get_feature_names_out()); coef = lr.coef_[0]
print("words pushing toward SUCCESS:", list(terms[np.argsort(coef)[-15:]][::-1]))
print("words pushing toward FAILURE:", list(terms[np.argsort(coef)[:15]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. One-hot encode categoricals (vocabulary from train)
# MAGIC
# MAGIC **What:** for region, major sector, environmental category, and document type: categories seen ≥ 30 times in train get their own 0/1 column; everything else (including unseen future values) becomes `other`.
# MAGIC
# MAGIC **Why:** a fixed, train-derived vocabulary keeps columns identical for train, test, and scoring — a model can't handle a column that suddenly appears in production.

# COMMAND ----------

onehot_cols = []
for c in CATEGORICALS:
    vc = pdf.loc[pdf["split"] == "train", c].value_counts()
    keep = set(vc[vc >= MIN_CAT_COUNT].index)
    vals = pdf[c].where(pdf[c].isin(keep), "other").fillna("other")
    d = pd.get_dummies(vals, prefix=c, dtype=float)
    d.columns = [str(x).lower().replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_")
                 for x in d.columns]
    onehot_cols += list(d.columns)
    pdf = pd.concat([pdf, d], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature groups
# MAGIC
# MAGIC **What:** labels every feature as `structured` or `text` and saves the mapping to `gold.feature_groups`.
# MAGIC
# MAGIC **Why:** the ablation in `06_compare` (structured-only vs text-only vs combined) needs a single, explicit definition of which columns belong to which family.

# COMMAND ----------

STRUCTURED = ["log_commitment", "log_project_cost", "grant_musd", "ida_share", "cofinancing_ratio",
              "n_sectors", "n_themes", "is_multi_country", "is_additional_financing",
              "prep_months", "appraisal_to_approval_months", "approved_in_june",
              "log_gdp_pc", "gdp_growth_pct", "inflation_pct", "ext_debt_pct_gni", "health_exp_pct_gdp",
              "educ_exp_pct_gdp", "log_population", "trade_pct_gdp", "fdi_pct_gdp",
              "wgi_gov_effectiveness", "wgi_regulatory_quality", "wgi_rule_of_law",
              "wgi_control_corruption", "wgi_political_stability", "wgi_voice_accountability",
              "country_prior_n", "country_prior_success", "sector_prior_n", "sector_prior_success",
              "region_prior_n", "region_prior_success"]
STRUCTURED += [c for c in onehot_cols if not c.startswith("doc_type_")]
TEXT = (["has_text", "n_words", "avg_sentence_words", "pct_numeric_tokens", "flesch_reading_ease", "doc_lead_days",
         "text_lr_score"]
        + [c for c in pdf.columns if c.startswith("kw_")] + pc_cols
        + [c for c in onehot_cols if c.startswith("doc_type_")])

groups = pd.DataFrame([(f, "structured") for f in STRUCTURED] + [(f, "text") for f in TEXT],
                      columns=["feature", "feature_group"])
spark.createDataFrame(groups).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.gold.feature_groups")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write the feature table with a primary key
# MAGIC
# MAGIC **What:** keeps ID/meta columns + all features, writes `gold.project_features`, and declares `project_id` as its primary key.
# MAGIC
# MAGIC **Why:** in Unity Catalog a Delta table with a primary key is a **feature table** — it shows up in the Features UI, can be looked up by key at training and scoring time, and documents that there is exactly one row per project. The table is dropped and recreated each run so the constraint is always applied to fresh data.

# COMMAND ----------

META = ["project_id", "project_name", "country_name", "approval_date", "approval_fy", "split", "label", "ieg_outcome"]
out = pdf[META + STRUCTURED + TEXT].copy()
out[STRUCTURED + TEXT] = out[STRUCTURED + TEXT].astype("float64")
out["label"] = out["label"].astype("float64")
assert out["project_id"].is_unique

table = f"{CATALOG}.gold.project_features"
spark.sql(f"DROP TABLE IF EXISTS {table}")
spark.createDataFrame(out).write.saveAsTable(table)
spark.sql(f"ALTER TABLE {table} ALTER COLUMN project_id SET NOT NULL")
spark.sql(f"ALTER TABLE {table} ADD CONSTRAINT project_features_pk PRIMARY KEY (project_id)")
spark.sql(f"COMMENT ON TABLE {table} IS 'One row per project: leakage-safe structured + text features, label, and time-based split'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Checks and summary
# MAGIC
# MAGIC **What:** rows and success rate per split, and how complete each feature family is.
# MAGIC
# MAGIC **Why:** if the success rate differs a lot between train and test, the metrics need that context; low text coverage in a split would weaken the text-vs-structured comparison.

# COMMAND ----------

g = spark.table(table)
display(g.groupBy("split").agg(F.count("*").alias("rows"), F.round(F.avg("label"), 3).alias("success_rate"),
                               F.round(F.avg("has_text"), 3).alias("has_text"),
                               F.round(F.avg(F.col("wgi_gov_effectiveness").isNotNull().cast("int")), 3).alias("has_wgi")))
summary = {
    "rows": g.count(),
    "splits": {r["split"]: r["count"] for r in g.groupBy("split").count().collect()},
    "n_structured": len(STRUCTURED), "n_text": len(TEXT),
    "pca_explained": round(float(pca.explained_variance_ratio_.sum()), 3),
}
print(summary)
dbutils.notebook.exit(json.dumps(summary))
