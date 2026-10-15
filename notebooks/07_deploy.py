# Databricks notebook source
# MAGIC %md
# MAGIC # 07_deploy — register the champion and batch-score active projects
# MAGIC
# MAGIC 1. **Pick** the champion feature set by **validation** AUC (never by test AUC — choosing on test would make the test score optimistic).
# MAGIC 2. **Register** that model in the Unity Catalog model registry as `dbw_wb_outcome.ml.project_outcome_xgb` and point the alias **`champion`** at the new version.
# MAGIC 3. **Batch-score** currently active projects (approved FY2019+, not yet rated) into `gold.active_project_scores`.
# MAGIC
# MAGIC **Why batch, not a serving endpoint:** projects are approved a few times a week and nobody needs a prediction in milliseconds; a scheduled batch job costs minutes of cluster time, while an always-on endpoint bills around the clock.
# MAGIC
# MAGIC **Attach to:** `wb-ml-single`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings
# MAGIC
# MAGIC **What:** registry location (`databricks-uc`), model name, experiment path.
# MAGIC
# MAGIC **Why:** setting the registry URI to Unity Catalog makes the model a governed UC object — versioned, permissioned, and lineage-linked to the tables it was trained on.

# COMMAND ----------

import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as F

CATALOG = "dbw_wb_outcome"
MODEL_NAME = f"{CATALOG}.ml.project_outcome_xgb"
ME = spark.sql("select current_user()").first()[0]
EXPERIMENT = f"/Users/{ME}/wb-outcome/wb_outcome_experiment"
from databricks.sdk import WorkspaceClient
WorkspaceClient().workspace.mkdirs(f"/Users/{ME}/wb-outcome")   # the experiment's parent folder must exist
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Choose the champion by validation AUC
# MAGIC
# MAGIC **What:** reads the latest parent run of each feature set and picks the highest `val_auc`.
# MAGIC
# MAGIC **Why:** model selection is part of training, so it must use validation data. The test score then stays an honest estimate of how the chosen model will do on future projects.

# COMMAND ----------

runs = mlflow.search_runs(experiment_names=[EXPERIMENT], filter_string="tags.stage = 'ablation'",
                          order_by=["start_time DESC"])
runs = runs[runs["tags.mlflow.parentRunId"].isna()] if "tags.mlflow.parentRunId" in runs else runs
latest = runs.drop_duplicates("tags.feature_set")
best = latest.sort_values("metrics.val_auc", ascending=False).iloc[0]
print(latest[["tags.feature_set", "metrics.val_auc", "metrics.test_auc"]].to_string(index=False))
print("champion:", best["tags.feature_set"], "| run", best["run_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Register in Unity Catalog and set the `champion` alias
# MAGIC
# MAGIC **What:** registers the run's logged model as a new version of `ml.project_outcome_xgb`, adds a description and tags, and moves the `champion` alias to it.
# MAGIC
# MAGIC **Why:** consumers load `models:/…@champion`, never a hard-coded version number. Promoting a new model = moving the alias; rolling back = moving it back. No consumer code changes.

# COMMAND ----------

mv = mlflow.register_model(best["tags.model_uri"], MODEL_NAME)
client.update_registered_model(MODEL_NAME, description=(
    "Average of 3 independently tuned XGBoost classifiers (MLflow pyfunc): "
    "P(World Bank project rated Moderately Satisfactory or better by IEG). "
    "Features known at Board approval only: project attributes, country indicators lagged 1 year, "
    "and pre-approval document text. Trained FY1995-2011, tested FY2012+."))
client.update_model_version(MODEL_NAME, mv.version, description=(
    f"feature_set={best['tags.feature_set']}; val_auc={best['metrics.val_auc']:.3f}; "
    f"test_auc={best['metrics.test_auc']:.3f}; source run {best['run_id']}"))
for k, v in {"feature_set": best["tags.feature_set"], "val_auc": f"{best['metrics.val_auc']:.4f}",
             "test_auc": f"{best['metrics.test_auc']:.4f}"}.items():
    client.set_model_version_tag(MODEL_NAME, mv.version, k, v)
client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
print(f"registered {MODEL_NAME} v{mv.version} -> @champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Batch-score active projects with `@champion`
# MAGIC
# MAGIC **What:** loads the model by alias (a pyfunc that averages 3 tuned XGBoost models), selects exactly the feature columns it stores, predicts P(success) for every `split = 'score'` project, and ranks them by risk of failure. The top 20% are flagged `High` risk.
# MAGIC
# MAGIC **Why:** reading the column list from the model itself guarantees training/scoring consistency. A ranked list with a risk band is what a portfolio manager would actually act on — "review these 20% first".

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
feature_cols = model.unwrap_python_model().features

score = spark.table(f"{CATALOG}.gold.project_features").filter("split = 'score'").toPandas()
score["prob_success"] = np.asarray(model.predict(score[feature_cols].astype("float64")))
score["risk_of_failure"] = 1 - score["prob_success"]
score["risk_rank"] = score["risk_of_failure"].rank(ascending=False, method="first").astype(int)
cut = score["risk_of_failure"].quantile(0.8)
score["risk_band"] = (score["risk_of_failure"] >= cut).map({True: "High", False: "Normal"})
score["model_version"] = int(mv.version)
score["scored_at"] = datetime.now(timezone.utc)

out = score[["project_id", "project_name", "country_name", "approval_date", "approval_fy",
             "prob_success", "risk_of_failure", "risk_rank", "risk_band", "model_version", "scored_at"]]
spark.createDataFrame(out).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.gold.active_project_scores")

display(spark.table(f"{CATALOG}.gold.active_project_scores").orderBy("risk_rank").limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Summary
# MAGIC
# MAGIC **What:** the registered version and scoring counts as the notebook exit value.
# MAGIC
# MAGIC **Why:** lets the scheduled workflow report what was deployed without opening the notebook.

# COMMAND ----------

summary = {"model": MODEL_NAME, "version": int(mv.version), "feature_set": best["tags.feature_set"],
           "val_auc": round(float(best["metrics.val_auc"]), 4), "test_auc": round(float(best["metrics.test_auc"]), 4),
           "scored_projects": len(out), "high_risk": int((out.risk_band == "High").sum())}
print(summary)
dbutils.notebook.exit(json.dumps(summary))
