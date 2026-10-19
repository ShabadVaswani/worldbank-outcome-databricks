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
# MAGIC Two methodological lessons:
# MAGIC - **A single tuning run can manufacture a "significant" result.** The v2 gain disappeared on a trivial re-run; averaging independently tuned models exposed it.
# MAGIC - **Stacking leaks through out-of-fold scores if the folds ignore time.** A 2003 project scored by a model that saw 2009 projects inflated validation by ~0.05 (text model's out-of-fold AUC 0.654 random-folds vs 0.542 time-respecting).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture (medallion on Delta Lake + Unity Catalog `dbw_wb_outcome`)
# MAGIC
# MAGIC ```
# MAGIC  SOURCES                        BRONZE (raw)                  SILVER (clean, leakage-safe) GOLD (model-ready)
# MAGIC  IEG ratings API             -> ieg_ratings                -> labels ---------------------+
# MAGIC  Projects API v3             -> projects                   -> projects -------------------+
# MAGIC  Indicators API (WDI + WGI)  -> indicators                 -> country_year ---------------+-> project_features
# MAGIC  Documents API (PAD/SAR/PID) -> doc_metadata, doc_manifest                                |   (PK project_id)
# MAGIC                                 raw_docs volume (.txt)     -> doc_text_features ----------+   + feature_groups
# MAGIC                                                               doc_chunks, doc_embeddings -+
# MAGIC
# MAGIC  05_train   XGBoost + Optuna (rolling CV) x3 seeds x3 feature sets -> MLflow -> gold.test_predictions
# MAGIC  06_compare paired bootstrap, ROC/PR curves -> gold.model_comparison(_gains)
# MAGIC  07_deploy  UC model ml.project_outcome_xgb @champion -> gold.active_project_scores
# MAGIC ```
# MAGIC
# MAGIC Orchestration: Workflow **`wb-outcome-pipeline`** (Jobs & Pipelines): ingest → (ingest_docs → unstructured_silver) ∥ structured_silver → gold_features → train → compare → deploy, on cluster `wb-ml-single`. Quarterly schedule exists but is **paused**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## How the problem was worked through
# MAGIC
# MAGIC The version table above shows what changed; this is the reasoning behind each decision.
# MAGIC
# MAGIC **Step 1 — Framing**
# MAGIC - **Question:** on the day the Board approves a project, can we predict whether IEG will later rate it Moderately Satisfactory or better?
# MAGIC - **What we found:** the main risk is leakage, because many IEG and project fields are written *after* approval.
# MAGIC - **Decision:** use only pre-approval information. Post-hoc IEG columns are dropped, indicators come from approval year − 1, only pre-approval document types are used, and track record uses only ratings published before approval. The split is by time: train FY1995–2011, test FY2012+ (2,217 test projects).
# MAGIC
# MAGIC **Step 2 — The first results looked like text helped**
# MAGIC - **Question:** does adding document text improve the prediction?
# MAGIC - **What we found:** v2 showed a "significant" +0.035 AUC gain from text. Re-running with 17 extra documents (v3) cut it to +0.017 (n.s.).
# MAGIC - **Decision:** treat a single tuning run as noisy. Each model became the average of 3 independently tuned XGBoost models (v4), and the text gain disappeared (−0.007, n.s.).
# MAGIC
# MAGIC **Step 3 — A leak in stacking**
# MAGIC - **Question:** why did the text models' validation AUC run about 0.05 above their test AUC?
# MAGIC - **What we found:** the supervised TF-IDF text score used random out-of-fold predictions, so a 2003 project could be scored by a model that had seen 2009 projects. Out-of-fold AUC was 0.654 with random folds versus 0.542 when time-respecting.
# MAGIC - **Decision:** out-of-fold scores must respect time (v6). Validation and test AUC agreed afterwards.
# MAGIC
# MAGIC **Step 4 — Checking every feature**
# MAGIC - **Question:** do all features hold sensible values?
# MAGIC - **What we found:** the box plot in the EDA showed "Flesch reading ease" at around −20,000 (valid values are roughly 0–100). It was secretly a proxy for document length.
# MAGIC - **Decision:** fix it, even though it moved AUC by only about 0.002 (v7, final).
# MAGIC
# MAGIC **Step 5 — Evaluating the final models**
# MAGIC - **Question:** does text add anything on top of the structured data?
# MAGIC - **What we found:** test AUC is 0.602 [0.573, 0.630] for structured, 0.576 for text and 0.589 for combined. The paired bootstrap for combined − structured is −0.013 [−0.032, +0.007].
# MAGIC - **Decision / interpretation:** approval-day information predicts only modestly, since outcomes are largely decided during implementation. Text carries real signal on its own (PR-AUC 0.267 vs 0.200 random) but mostly the same information as the structured data, so it is a substitute, not an add-on.
# MAGIC
# MAGIC **Step 6 — Is it useful anyway?**
# MAGIC - **Question:** even with a modest AUC, does the ranking separate risky projects from safe ones?
# MAGIC - **What we found:** on test projects, the riskiest 10% failed about 3× as often as the safest 10% (32% vs 11%), and the riskiest 20% contain 27.5% of all failures.
# MAGIC - **Decision:** the champion is the structured model, chosen by validation AUC (not test) and registered as `ml.project_outcome_xgb@champion` (v7). It batch-scores the 2,505 active projects into `gold.active_project_scores`, with the top 20% flagged High risk as a "review these first" list.
# MAGIC
# MAGIC **Step 7 — Next step**
# MAGIC - **Question:** what would improve the predictions most?
# MAGIC - **What we found:** the biggest gain should come from a during-implementation model that uses supervision ratings, disbursement pace and closing-date extensions to predict around year 2. That answers a different question: which running projects need rescue?
# MAGIC - **Decision:** before that, add a champion/challenger gate and data-drift monitoring on the feature table and the score distribution.
# MAGIC
# MAGIC **Step 8 — Extension: using the documents to explain and question the results** (`09`–`12`)
# MAGIC - **Question:** text added nothing to the *prediction*. Can the same documents still help a reviewer understand a flagged project, with evidence?
# MAGIC - **What we found:** a retrieval-augmented assistant over 50,000 chunks from 453 project documents answers questions with citations. Hybrid search ranks the right passage higher than vector search (MRR 0.58 → 0.74). On a 50-question evaluation, every answer was grounded in the retrieved text, all citations were valid, all 5 unanswerable questions got "not found" and all 5 off-topic or injection attempts were refused. The LLM judge scored correctness at 67.5%, but a human review of the 13 answers it rejected found 11 correct and 2 partly correct (one reference answer was itself wrong): 38 of 40 answers are fully correct.
# MAGIC - **Decision:** guardrails in layers (an LLM input guard, evidence treated as data, citation checks in code, abstention), reasoning mode off (no quality gain, about 8× slower), and per-project notes for the 20 riskiest active projects that combine the model's top risk drivers with quotes from the project's own documents (17 of 20 judged fully grounded).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook guide
# MAGIC
# MAGIC | # | Notebook | What it does / key techniques |
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
# MAGIC | 9 | `09_rag_index` | Structure-aware chunking, embeddings in Delta, Vector Search index (FAISS as a drop-in backend), hybrid retrieval |
# MAGIC | 10 | `10_rag_assistant` | RAG with citations, LLM input guard, abstention, JSON output validation, citation checks, MLflow tracing |
# MAGIC | 11 | `11_rag_evaluation` | Generated test set, hit rate and MRR, LLM judges from a different model family, code checks, pre-registered thinking on/off test |
# MAGIC | 12 | `12_flagged_project_notes` | SHAP risk drivers + retrieved evidence → cited notes for flagged projects, groundedness check |
# MAGIC
# MAGIC Where to look: **Catalog** → `dbw_wb_outcome` (tables + Lineage tab) · **Experiments** → `wb-outcome/wb_outcome_experiment` (the latest `xgb_*` runs are final; earlier ones are the versions above) · **Models** → `ml.project_outcome_xgb` (v7 = `@champion`) · **Report** → `08_results_report` (14 charts; outputs in `executed/08_results_report`; `REPORT.md` and figures in the `gold.reports` volume) · **LLM traces** → `wb-outcome/wb_rag_experiment` (every assistant answer, step by step).

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
# MAGIC ## Cost controls
# MAGIC `wb-ml-single` auto-terminates after 10 idle minutes, the workflow schedule is paused, and scoring is batch (no always-on model-serving endpoint).
