# Databricks notebook source
# MAGIC %md
# MAGIC # World Bank Project Outcome Predictor
# MAGIC
# MAGIC **Question:** on the day the World Bank Board approves a project, can we predict whether it will later be rated a success by the Independent Evaluation Group (IEG outcome *Moderately Satisfactory or better*)?
# MAGIC
# MAGIC **Data:** 12,475 IEG-rated projects; 7,002 approved FY1995+ are modelled (train FY1995–2011: 4,785 · test FY2012+: 2,217). 15 country indicators (WDI + WGI), project attributes from the Projects API, and the text of 8,318 pre-approval documents (PAD / Staff Appraisal Report / Program Document / PID).
# MAGIC
# MAGIC **Final results** (test = projects approved FY2012+, never used for training or tuning; each model = average of 3 independently tuned XGBoost models):
# MAGIC
# MAGIC | Model | Validation AUC | **Test AUC** (95% CI) | PR-AUC on failures (random = 0.200) | Brier |
# MAGIC |---|---|---|---|---|
# MAGIC | **Structured** (project + country + track record) | 0.613 | **0.602** [0.573, 0.630] | **0.269** | 0.160 |
# MAGIC | Text (document features, embeddings, TF-IDF score) | 0.579 | 0.576 [0.547, 0.607] | 0.267 | 0.162 |
# MAGIC | Combined | 0.607 | 0.589 [0.559, 0.619] | 0.262 | 0.163 |
# MAGIC
# MAGIC Paired bootstrap, combined − structured: **−0.013 AUC, 95% CI [−0.032, +0.007]** → no significant gain from adding text.
# MAGIC
# MAGIC **What this means**
# MAGIC 1. Approval-day information predicts outcomes only modestly (AUC ≈ 0.60). That's expected: outcomes are largely decided *during* implementation, which an approval-day model can't see.
# MAGIC 2. **Text alone carries real signal** (AUC 0.58, PR-AUC 0.267 vs 0.200 random), and the TF-IDF model's words are interpretable — "civil service", "institutional framework", "private sector", "HIV/AIDS" lean toward failure; "roads", "rural", "maintenance", "emergency" toward success (institutional-reform projects are known to be harder than infrastructure).
# MAGIC 3. But it mostly carries the **same information** as the structured data (sector, region, country track record), so combining them doesn't help. That makes text a **substitute, not an add-on**. It isn't useless: on its own it's almost as good at finding failures (PR-AUC 0.267 vs 0.269), and it may matter most where structured data is thin (untested here).
# MAGIC 4. Strongest structured drivers (SHAP): region and country **track record**, **preparation time**, project size, GDP growth.
# MAGIC 5. The deployed champion (`ml.project_outcome_xgb@champion`, chosen by validation AUC) is the structured model; it ranks 2,505 active projects into `gold.active_project_scores`, top 20% flagged High risk.

# COMMAND ----------

# MAGIC %md
# MAGIC ## How the result evolved — and why the final numbers are trustworthy
# MAGIC
# MAGIC Every version below ran end to end; each change fixed a *method* problem, not a score we disliked.
# MAGIC
# MAGIC | Version | Change | Combined test AUC | Text gain vs structured | What we learned |
# MAGIC |---|---|---|---|---|
# MAGIC | v1 | single validation window (FY2008–11), one Optuna run | 0.588 | +0.016 (n.s.) | Models stopped at the minimum tree count: weak signal, noisy tuning |
# MAGIC | v2 | + country/sector/region track record, rolling-origin CV | 0.608 | **+0.035 (significant!)** | Looked like the headline result… |
# MAGIC | v3 | same code, re-run with 17 extra documents | 0.590 | +0.017 (n.s.) | …but it moved with a trivial data change → **tuning variance**, not signal |
# MAGIC | v4 | each model = average of 3 independently tuned models | 0.595 | −0.007 (n.s.) | Stable estimate: structured 0.602, text adds nothing |
# MAGIC | v5 | + supervised TF-IDF text score, *random* out-of-fold | 0.591 | −0.011 (n.s.) | Validation AUC of text models ~0.05 above test → **leak** |
# MAGIC | v6 | out-of-fold made **time-respecting** | 0.591 | −0.011 (n.s.) | Validation ≈ test for every model → leak gone |
# MAGIC | **v7 (final)** | readability feature fixed (it was a hidden length proxy, values ≈ −20,000) | 0.589 | −0.013 (n.s.) | Plot every feature: the bug was spotted in the `08_results_report` box plot |
# MAGIC
# MAGIC Two lessons worth telling in an interview:
# MAGIC - **A single tuning run can manufacture a "significant" result.** The v2 gain disappeared on a trivial re-run; averaging independently tuned models exposed it.
# MAGIC - **Stacking leaks through out-of-fold scores if the folds ignore time.** A 2003 project scored by a model that saw 2009 projects inflated validation by ~0.05 (text model's out-of-fold AUC 0.654 random-folds vs 0.542 time-respecting).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture (medallion on Delta Lake + Unity Catalog `dbw_wb_outcome`)
# MAGIC
# MAGIC ```
# MAGIC  SOURCES                         BRONZE (raw)               SILVER (clean, leakage-safe)       GOLD (model-ready)
# MAGIC  IEG ratings API ─────────────▶ ieg_ratings ─────────────▶ labels ──────────────┐
# MAGIC  Projects API v3 ────────────▶ projects ────────────────▶ projects ────────────┤
# MAGIC  Indicators API (WDI + WGI) ──▶ indicators ──────────────▶ country_year ────────┼──▶ project_features (PK project_id)
# MAGIC  Documents API (PAD/SAR/PID) ─▶ doc_metadata, doc_manifest                      │      + feature_groups
# MAGIC                                 raw_docs volume (.txt) ──▶ doc_text_features ───┤
# MAGIC                                                           doc_chunks, doc_embeddings ┘
# MAGIC
# MAGIC  05_train   XGBoost + Optuna (rolling CV) ×3 seeds ×3 feature sets → MLflow → gold.test_predictions
# MAGIC  06_compare paired bootstrap, ROC/PR curves → gold.model_comparison(_gains)
# MAGIC  07_deploy  UC model ml.project_outcome_xgb @champion → gold.active_project_scores
# MAGIC ```
# MAGIC
# MAGIC Orchestration: Workflow **`wb-outcome-pipeline`** (Jobs & Pipelines): ingest → (ingest_docs → unstructured_silver) ∥ structured_silver → gold_features → train → compare → deploy, on cluster `wb-ml-single`. Quarterly schedule exists but is **paused**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to run it yourself
# MAGIC
# MAGIC **Option A — whole pipeline:** Jobs & Pipelines → `wb-outcome-pipeline` → **Run now** (~1 hour including cluster start: documents are cached and embeddings are incremental; a from-scratch run is ~2 hours, mostly embeddings).
# MAGIC
# MAGIC **Option B — notebook by notebook (best for learning):** open each in order, attach `wb-ml-single`, **Run all**, and read the *What / Why* above every cell.
# MAGIC
# MAGIC | # | Notebook | Concepts you'll learn |
# MAGIC |---|---|---|
# MAGIC | 0 | `00_setup` | Unity Catalog (catalog → schema → table/volume), medallion layers, idempotency |
# MAGIC | 1 | `01_ingest` | API paging, raw snapshots, Delta tables, lineage columns |
# MAGIC | 1b | `01b_ingest_docs` | Leakage-safe document selection, parallel resumable downloads |
# MAGIC | 2 | `02_structured_silver` | Label definition, dropping post-hoc columns, forward-only gap filling |
# MAGIC | 3 | `03_unstructured_silver` | Text cleaning, keyword rates, readability, sentence embeddings, incremental processing |
# MAGIC | 4 | `04_gold_features` | Time-based split, lagged joins, track-record features, PCA fit on train only, TF-IDF stacking with time-respecting out-of-fold, feature tables |
# MAGIC | 5 | `05_train` | XGBoost, early stopping, Optuna/TPE, rolling-origin CV, seed ensembles, MLflow nested runs, pyfunc, calibration, SHAP |
# MAGIC | 6 | `06_compare` | Ablation, ROC vs PR curves, paired bootstrap significance |
# MAGIC | 7 | `07_deploy` | UC model registry, aliases (`@champion`), batch vs real-time serving |
# MAGIC
# MAGIC Where to look: **Catalog** → `dbw_wb_outcome` (tables + Lineage tab) · **Experiments** → `wb-outcome/wb_outcome_experiment` (the latest `xgb_*` runs are final; earlier ones are the versions above) · **Models** → `ml.project_outcome_xgb` (v7 = `@champion`) · **Report** → `08_results_report` (14 charts, `REPORT.md` in the `gold.reports` volume).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leakage controls
# MAGIC
# MAGIC | Risk | Control | Where |
# MAGIC |---|---|---|
# MAGIC | Text describing the outcome | Only pre-approval document types, dated on/before Board approval; completion reports (ICR/ICRR) never used | `01b` |
# MAGIC | Post-hoc IEG columns | Bank performance, quality ratings, closing dates, closing-year FCS status dropped | `02` |
# MAGIC | Revised project fields | Current closing date, status, effectiveness/ICR milestones, possibly-revised PDO not used | `02` |
# MAGIC | Macro data from implementation years | Indicators joined on approval year − 1; gaps filled forward only | `02`, `04` |
# MAGIC | Track record including the future | Only IEG ratings published before the approval FY | `04` |
# MAGIC | Preprocessing peeking at test | PCA, one-hot vocabulary, TF-IDF vocabulary fit on train rows only | `04` |
# MAGIC | Stacking leak | TF-IDF score for train rows from models trained on earlier years only | `04` |
# MAGIC | Tuning on test / random split | Time split; rolling-origin CV inside train; champion chosen by validation AUC | `05`, `07` |
# MAGIC
# MAGIC ## Limitations
# MAGIC - **Test-period selection bias:** only projects that already closed and were rated are in test; long recent projects are missing.
# MAGIC - **OCR noise** in 1990s Staff Appraisal Reports; **regional projects** have no single country; **embeddings use ~1,600 words** per document (CPU budget).
# MAGIC - **No implementation-time signals** by design (they'd be leakage for an approval-day model). A natural extension is a "during implementation" model using ISR ratings and disbursement speed, which should predict far better.
# MAGIC
# MAGIC ## Cost hygiene
# MAGIC `wb-ml-single` auto-terminates after 10 idle minutes; the workflow schedule is paused; no serving endpoint exists. When done, delete the Azure resource group `rg-wb-outcome` (removes everything).
