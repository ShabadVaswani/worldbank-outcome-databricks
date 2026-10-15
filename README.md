# World Bank Project Outcome Predictor

**Can we predict, on the day the World Bank Board approves a project, whether it will later be rated a success?**

An end-to-end ML pipeline on **Azure Databricks** that combines structured data (project attributes, country macro and governance indicators) with unstructured data (the text of ~8,300 pre-approval project documents) to predict the Independent Evaluation Group (IEG) outcome rating. Built with Delta Lake, Unity Catalog, MLflow, XGBoost, and Databricks Workflows.

## Results

Test set = **2,217 projects approved FY2012 or later**, never used for training or tuning. Each model is the average of 3 independently tuned XGBoost models.

| Model | Validation AUC | **Test AUC** (95% CI) | PR-AUC on failures (random = 0.200) |
|---|---|---|---|
| **Structured** — project + country + track record | 0.613 | **0.602** [0.573, 0.630] | **0.269** |
| Text — document features, embeddings, TF-IDF score | 0.582 | 0.578 [0.549, 0.610] | 0.257 |
| Combined | 0.605 | 0.591 [0.561, 0.621] | 0.255 |

Paired bootstrap, combined − structured: **−0.011 AUC, 95% CI [−0.030, +0.008]**.

**Findings**
- Approval-day information predicts outcomes only modestly (AUC ≈ 0.60). That's expected: outcomes are largely decided during implementation, which an approval-day model can't see.
- **The document text carries real signal on its own** (AUC 0.58; PR-AUC 0.257 vs 0.200 random), and it's interpretable: "civil service", "institutional framework", "private sector", "HIV/AIDS" lean toward failure; "roads", "rural", "maintenance", "emergency" toward success.
- **But it is redundant with the structured data** — adding it to the structured model gives no significant gain.
- Strongest structured drivers (SHAP): regional and country **track record** of earlier projects, **preparation time**, project size, GDP growth.

### How the result evolved (and why these numbers are trustworthy)

| Version | Change | Combined test AUC | Text gain vs structured |
|---|---|---|---|
| v1 | single validation window, one Optuna run | 0.588 | +0.016 (n.s.) |
| v2 | + track-record features, rolling-origin CV | 0.608 | **+0.035 (significant)** |
| v3 | same code, 17 more documents | 0.590 | +0.017 (n.s.) |
| v4 | each model = average of 3 independently tuned models | 0.595 | −0.007 (n.s.) |
| v5 | + supervised TF-IDF text score (random out-of-fold) | 0.591 | −0.011 (n.s.) |
| **v6** | out-of-fold made **time-respecting** | **0.591** | **−0.011 (n.s.)** |

Two lessons:
1. **A single tuning run can manufacture a "significant" result.** The v2 gain vanished on a trivial re-run; averaging independently tuned models exposed it as tuning variance.
2. **Stacking leaks if out-of-fold scores ignore time.** Scoring a 2003 project with a model that saw 2009 projects inflated validation by ~0.05 (TF-IDF out-of-fold AUC 0.654 with random folds vs 0.542 time-respecting).

## Data

| Source | API | Used for |
|---|---|---|
| IEG World Bank Project Performance Ratings (12,627 projects) | Finances One `DS00053` | Label: Moderately Satisfactory or better = success |
| World Bank Projects API v3 (28,184 projects) | `search.worldbank.org/api/v3/projects` | Commitment, sector, region, preparation milestones |
| World Development Indicators + Worldwide Governance Indicators (15 series) | `api.worldbank.org/v2` | Country context in the year **before** approval |
| Documents & Reports (PAD, Staff Appraisal Report, Program Document, PID) | `search.worldbank.org/api/v2/wds` | Text of the latest pre-approval document per project |

Modelled population: 7,002 rated projects approved FY1995+ (train FY1995–2011, test FY2012+), plus 2,505 active projects scored by the deployed model.

## Architecture

Medallion layout on Delta Lake, governed by Unity Catalog:

```
 SOURCES                         BRONZE (raw)               SILVER (clean, leakage-safe)       GOLD (model-ready)
 IEG ratings API ─────────────▶ ieg_ratings ─────────────▶ labels ──────────────┐
 Projects API v3 ────────────▶ projects ────────────────▶ projects ────────────┤
 Indicators API (WDI + WGI) ──▶ indicators ──────────────▶ country_year ────────┼──▶ project_features (PK project_id)
 Documents API (PAD/SAR/PID) ─▶ doc_metadata, doc_manifest                      │      + feature_groups
                                raw_docs volume (.txt) ──▶ doc_text_features ───┤
                                                          doc_chunks, doc_embeddings ┘

 05_train   XGBoost + Optuna (rolling-origin CV) ×3 seeds ×3 feature sets → MLflow
 06_compare paired bootstrap, ROC / PR curves
 07_deploy  Unity Catalog model registry, alias @champion → batch scores for active projects
```

Workflow DAG (`databricks.yml`): `ingest → {ingest_docs → unstructured_silver, structured_silver} → gold_features → train → compare → deploy`.

## Leakage controls

| Risk | Control |
|---|---|
| Text describing the outcome | Only pre-approval document types, dated on/before Board approval; completion reports never used |
| Post-hoc IEG fields | Bank performance, quality ratings, closing dates, closing-year fragility status dropped |
| Revised project fields | Current closing date, status, effectiveness/ICR milestones not used |
| Macro data from implementation years | Indicators from approval year − 1; gaps filled forward only |
| Track record including the future | Only IEG ratings published before the approval year |
| Preprocessing peeking at test | PCA, one-hot vocabulary, TF-IDF vocabulary fit on train only |
| Stacking leak | TF-IDF score for training rows from models trained on earlier years only |
| Tuning on test | Time-based split; rolling-origin CV inside train; champion chosen by validation AUC |

## Repository layout

```
notebooks/
  00_README.py               results, theory map, limitations (Databricks notebook)
  00_setup.py                Unity Catalog schemas + volume, smoke test
  01_ingest.py               IEG ratings, Projects API, indicators → bronze
  01b_ingest_docs.py         document catalog, leakage-safe selection, parallel download
  02_structured_silver.py    labels, project features, country-year table
  03_unstructured_silver.py  text cleaning, keyword/readability features, MiniLM embeddings (incremental)
  04_gold_features.py        joins, track record, PCA, TF-IDF stacking, feature table with PK
  05_train.py                XGBoost + Optuna, rolling-origin CV, 3-seed ensemble, MLflow, SHAP
  06_compare.py              ablation with paired bootstrap
  07_deploy.py               UC registry @champion, batch scoring
databricks.yml               Asset Bundle: 8-task workflow (schedule paused)
```

Every code cell is preceded by a **What / Why** explanation.

## Reproduce

1. **Workspace:** Azure Databricks (Premium or Trial), Unity Catalog enabled. Notebooks use the catalog `dbw_wb_outcome` — change the `CATALOG` constant at the top of each notebook if yours differs.
2. **Cluster:** single node, Databricks Runtime **18 LTS ML** (CPU), Dedicated access mode, e.g. `Standard_D4ds_v4`, Photon off.
3. **Setup:** run `notebooks/00_setup.py` once (creates `bronze` / `silver` / `gold` / `ml` schemas and the `raw_docs` volume).
4. **Deploy and run** with the [Databricks CLI](https://learn.microsoft.com/azure/databricks/dev-tools/cli/):
   ```bash
   databricks bundle deploy --var cluster_id=<your-cluster-id>
   databricks bundle run wb_outcome_pipeline --var cluster_id=<your-cluster-id>
   ```
   A from-scratch run takes ~2 hours on 4 CPU cores (mostly document embeddings); later runs reuse cached documents and embeddings (~1 hour).

## Limitations

- Only projects that have closed and been rated are in the test period (selection bias toward shorter projects).
- 1990s Staff Appraisal Reports are OCR text with noise; regional projects have no single country for macro features.
- Embeddings cover ~1,600 words per document (CPU budget).
- No implementation-time signals by design — a natural extension is a "during implementation" model using supervision ratings and disbursement speed.

## Tech stack

Azure Databricks · Delta Lake · Unity Catalog (volumes, feature table, model registry) · MLflow 3 (nested runs, pyfunc) · XGBoost 3 · Optuna · scikit-learn · sentence-transformers (all-MiniLM-L6-v2) · SHAP (TreeSHAP) · Databricks Workflows · Databricks Asset Bundles

Data © World Bank Group, licensed CC BY 4.0.
