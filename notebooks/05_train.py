# Databricks notebook source
# MAGIC %md
# MAGIC # 05_train — XGBoost + Optuna, tracked in MLflow
# MAGIC
# MAGIC Trains **three models with the same recipe**, one per feature set, so their scores are directly comparable:
# MAGIC
# MAGIC | Feature set | Columns |
# MAGIC |---|---|
# MAGIC | `structured` | project details, country macro/governance, country/sector/region track record |
# MAGIC | `text` | document features + embedding components |
# MAGIC | `combined` | both |
# MAGIC
# MAGIC **Recipe per feature set:** tune XGBoost with Optuna **3 independent times** (different random seeds), refit each winner on all training years, and **average the 3 models' probabilities**. That averaged model is what gets evaluated, logged, and deployed.
# MAGIC
# MAGIC Why 3 tuning repeats? A single Optuna run is noisy: a tiny data change can send it to different hyperparameters and move test AUC by ±0.02 — the same size as the effect we want to measure. Averaging independently tuned models cancels much of that noise (it's a small *ensemble*), and the spread between the 3 repeats is logged so you can see how stable the result is.
# MAGIC
# MAGIC **Metrics** (label 1 = success):
# MAGIC - **AUC** (ROC-AUC): probability a random successful project scores above a random failed one. 0.5 = coin flip.
# MAGIC - **PR-AUC on failures** (average precision with failure as the positive class): how well the model surfaces failing projects. Random ≈ the failure rate.
# MAGIC - **F1 / precision / recall on failures** at a threshold chosen on validation data.
# MAGIC - **Brier score + calibration curve**: do predicted probabilities match reality (does "70%" really mean 70%)?
# MAGIC
# MAGIC **Attach to:** `wb-ml-single`. Takes ~20 minutes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings
# MAGIC
# MAGIC **What:** experiment location, tuning budget, repeat seeds, and the validation folds.
# MAGIC
# MAGIC **Why:** tuning must not peek at the test set, so validation periods are carved out of train with **rolling-origin (expanding-window) cross-validation** — each fold fits on all earlier years and validates on the next block:
# MAGIC
# MAGIC | fold | fit on FY | validate on FY |
# MAGIC |---|---|---|
# MAGIC | 1 | 1995–2001 | 2002–2004 |
# MAGIC | 2 | 1995–2004 | 2005–2007 |
# MAGIC | 3 | 1995–2007 | 2008–2011 |
# MAGIC
# MAGIC Averaging AUC over three windows is far less noisy than one window, and every fold respects time order (past → future). The test set (FY2012+) is touched once, at the end.

# COMMAND ----------

import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow, optuna, sklearn, xgboost as xgb
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, brier_score_loss
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

CATALOG = "dbw_wb_outcome"
FOLDS = [(2002, 2004), (2005, 2007), (2008, 2011)]   # (validation start FY, validation end FY)
N_TRIALS = 30
REPEAT_SEEDS = [42, 7, 123]
ME = spark.sql("select current_user()").first()[0]
EXPERIMENT = f"/Users/{ME}/wb-outcome/wb_outcome_experiment"
from databricks.sdk import WorkspaceClient
WorkspaceClient().workspace.mkdirs(f"/Users/{ME}/wb-outcome")   # the experiment's parent folder must exist
mlflow.set_experiment(EXPERIMENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load features and splits
# MAGIC
# MAGIC **What:** reads `gold.project_features` and `gold.feature_groups`, builds the three feature lists, and builds the train/test sets and the 3 rolling folds.
# MAGIC
# MAGIC **Why:** one source table for every model guarantees the comparison is apples to apples — same rows, same labels, only the columns differ.

# COMMAND ----------

gold = spark.table(f"{CATALOG}.gold.project_features").toPandas()
groups = spark.table(f"{CATALOG}.gold.feature_groups").toPandas()
FEATURES = {
    "structured": groups.loc[groups.feature_group == "structured", "feature"].tolist(),
    "text": groups.loc[groups.feature_group == "text", "feature"].tolist(),
}
FEATURES["combined"] = FEATURES["structured"] + FEATURES["text"]

train = gold[gold.split == "train"].reset_index(drop=True)
test = gold[gold.split == "test"].reset_index(drop=True)
folds = [(train[train.approval_fy < a], train[train.approval_fy.between(a, b)]) for a, b in FOLDS]
print("folds (fit, val):", [(len(f), len(v)) for f, v in folds], "| train", len(train), "| test", len(test))
print("success rate  train:", round(train.label.mean(), 3), " test:", round(test.label.mean(), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Helpers
# MAGIC
# MAGIC **What:**
# MAGIC - `make_model` builds an XGBoost classifier; `cv_fit` trains it on each rolling fold.
# MAGIC - `evaluate` computes all metrics; `best_f1_threshold` picks the failure threshold; `plot_calibration` draws predicted vs observed success.
# MAGIC - `XGBEnsemble` is an MLflow **pyfunc** model that averages several XGBoost models' probabilities — so the averaged model can be registered and served like any other model.
# MAGIC
# MAGIC **Why:**
# MAGIC - **Early stopping:** trees are added until validation AUC stops improving for 50 rounds — the model picks its own size.
# MAGIC - **PR-AUC on failures** flips labels and scores (`1 - y`, `1 - p`) because failure is the minority class we want to find.
# MAGIC - **Missing values** (`NaN`) are fine: XGBoost learns which branch missing values should follow.
# MAGIC - **pyfunc** is MLflow's generic "anything with a `predict`" format; the ensemble stores its own feature list so scoring always uses the right columns in the right order.

# COMMAND ----------

def make_model(params, seed, n_estimators=2000, early_stopping=True):
    return xgb.XGBClassifier(
        n_estimators=n_estimators, tree_method="hist", eval_metric="auc", n_jobs=4, random_state=seed,
        early_stopping_rounds=50 if early_stopping else None, **params)

def cv_fit(params, cols, seed):
    """Train on each rolling fold; return mean val AUC, mean best iteration, pooled validation (y, p), per-fold AUCs."""
    aucs, iters, ys, ps = [], [], [], []
    for f, v in folds:
        m = make_model(params, seed).fit(f[cols], f.label.values, eval_set=[(v[cols], v.label.values)], verbose=False)
        p = m.predict_proba(v[cols])[:, 1]
        aucs.append(roc_auc_score(v.label.values, p)); iters.append(m.best_iteration)
        ys.append(v.label.values); ps.append(p)
    return float(np.mean(aucs)), float(np.mean(iters)), np.concatenate(ys), np.concatenate(ps), aucs

def evaluate(y, p, threshold):
    yf, pf = 1 - y, 1 - p
    pred_fail = (pf >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y, p),
        "pr_auc_failure": average_precision_score(yf, pf),
        "failure_rate": float(yf.mean()),
        "f1_failure": f1_score(yf, pred_fail, zero_division=0),
        "precision_failure": precision_score(yf, pred_fail, zero_division=0),
        "recall_failure": recall_score(yf, pred_fail, zero_division=0),
        "brier": brier_score_loss(y, p),
    }

def best_f1_threshold(y, p):
    yf, pf = 1 - y, 1 - p
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(yf, (pf >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])

def plot_calibration(y, p, title):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set_xlabel("predicted P(success)"); ax.set_ylabel("observed success rate"); ax.set_title(title)
    ax.legend()
    return fig

class XGBEnsemble(mlflow.pyfunc.PythonModel):
    """Averages P(success) over several XGBoost classifiers trained on the same feature list."""
    def __init__(self, models, features):
        self.models, self.features = models, list(features)

    def predict(self, context, model_input, params=None):
        X = model_input[self.features].astype("float64")
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tune ×3, refit, average, evaluate, log — for each feature set
# MAGIC
# MAGIC **What:** for each feature set, inside one MLflow parent run:
# MAGIC 1. For each of 3 seeds, **Optuna** tries 30 hyperparameter combinations (depth, learning rate, regularisation, sampling); each trial is scored by **mean validation AUC over the 3 rolling folds** and logged as a nested run.
# MAGIC 2. Each repeat's winner is **refit on all training years** (tree count = average early-stopping point × 1.1).
# MAGIC 3. The 3 refit models are **averaged**. The failure threshold is chosen on the averaged **pooled validation** predictions; the averaged model is scored **once** on test.
# MAGIC 4. Logs: per-repeat test AUC (stability), ensemble metrics + bootstrap CI, calibration plot, SHAP importance, and the ensemble as a pyfunc model with a signature (required by the Unity Catalog registry).
# MAGIC
# MAGIC **Why:** Optuna's TPE sampler learns which regions of the search space work and focuses there — better than random search for the same budget. **SHAP** splits each prediction into per-feature contributions; we use XGBoost's exact TreeSHAP (`pred_contribs=True`), averaged over the 3 models.

# COMMAND ----------

results, test_preds = {}, []

for fs, cols in FEATURES.items():
    with mlflow.start_run(run_name=f"xgb_{fs}") as parent:
        mlflow.set_tags({"feature_set": fs, "model_family": "xgboost_ensemble", "stage": "ablation"})
        mlflow.log_params({"feature_set": fs, "n_features": len(cols), "n_trials_per_repeat": N_TRIALS,
                           "repeat_seeds": str(REPEAT_SEEDS),
                           "cv": "rolling-origin: val FY2002-04, 2005-07, 2008-11", "test_fy": "2012+"})
        models, val_aucs, val_ps, rep_test_aucs = [], [], [], []

        for seed in REPEAT_SEEDS:
            def objective(trial):
                params = {
                    "max_depth": trial.suggest_int("max_depth", 2, 6),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "min_child_weight": trial.suggest_float("min_child_weight", 1, 30, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                }
                with mlflow.start_run(run_name=f"seed{seed}_trial{trial.number:02d}", nested=True):
                    auc, it, _, _, fold_aucs = cv_fit(params, cols, seed)
                    mlflow.log_params({**params, "repeat_seed": seed})
                    mlflow.log_metrics({"val_auc": auc, "best_iteration": it,
                                        **{f"val_auc_fold{i+1}": a for i, a in enumerate(fold_aucs)}})
                    trial.set_user_attr("best_iteration", it)
                return auc

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(objective, n_trials=N_TRIALS)
            best = study.best_params
            n_trees = max(30, int(study.best_trial.user_attrs["best_iteration"] * 1.1))
            _, _, y_val, p_val, _ = cv_fit(best, cols, seed)
            m = make_model(best, seed, n_estimators=n_trees, early_stopping=False).fit(train[cols], train.label.values)
            models.append(m); val_aucs.append(study.best_value); val_ps.append(p_val)
            rep_test_aucs.append(roc_auc_score(test.label.values, m.predict_proba(test[cols])[:, 1]))
            mlflow.log_params({f"seed{seed}_{k}": v for k, v in {**best, "n_trees": n_trees}.items()})
            mlflow.log_metrics({f"seed{seed}_val_auc": study.best_value, f"seed{seed}_test_auc": rep_test_aucs[-1]})

        ensemble = XGBEnsemble(models, cols)
        p_val_ens = np.mean(val_ps, axis=0)
        threshold = best_f1_threshold(y_val, p_val_ens)
        y_t = test.label.values
        p_test = ensemble.predict(None, test)
        metrics = evaluate(y_t, p_test, threshold)

        rng = np.random.default_rng(42)
        boots = []
        for _ in range(1000):
            idx = rng.integers(0, len(y_t), len(y_t))
            if 0 < y_t[idx].sum() < len(idx):
                boots.append(roc_auc_score(y_t[idx], p_test[idx]))
        metrics.update(auc_ci_low=float(np.percentile(boots, 2.5)), auc_ci_high=float(np.percentile(boots, 97.5)),
                       repeat_test_auc_min=float(min(rep_test_aucs)), repeat_test_auc_max=float(max(rep_test_aucs)))
        val_auc = float(np.mean(val_aucs))

        mlflow.log_param("failure_threshold", round(threshold, 3))
        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        fig = plot_calibration(y_t, p_test, f"Calibration — {fs} (test)")
        mlflow.log_figure(fig, "calibration_test.png"); plt.close(fig)

        dm = xgb.DMatrix(test[cols])
        contribs = np.mean([m.get_booster().predict(dm, pred_contribs=True)[:, :-1] for m in models], axis=0)
        imp = pd.Series(np.abs(contribs).mean(axis=0), index=cols).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 8))
        imp.head(20)[::-1].plot.barh(ax=ax); ax.set_title(f"Mean |SHAP| — top 20 ({fs})"); ax.set_xlabel("mean |SHAP| (log-odds)")
        fig.tight_layout(); mlflow.log_figure(fig, "shap_top20.png"); plt.close(fig)
        mlflow.log_text(imp.to_string(), "shap_importance.txt")

        example = test[cols].head(5).astype("float64")
        sig = infer_signature(example, ensemble.predict(None, example))
        info = mlflow.pyfunc.log_model(
            name="model", python_model=ensemble, signature=sig, input_example=example,
            pip_requirements=[f"xgboost=={xgb.__version__}", f"scikit-learn=={sklearn.__version__}",
                              f"pandas=={pd.__version__}", f"numpy=={np.__version__}"])
        mlflow.set_tag("model_uri", info.model_uri)

        metrics.update(val_auc=val_auc, failure_threshold=threshold,
                       repeat_test_aucs=[round(a, 4) for a in rep_test_aucs], top_shap=list(imp.head(8).index))
        results[fs] = {**metrics, "run_id": parent.info.run_id, "model_uri": info.model_uri}
        test_preds.append(pd.DataFrame({"project_id": test.project_id, "approval_fy": test.approval_fy,
                                        "label": y_t, "feature_set": fs, "prob_success": p_test}))
        print(f"{fs:10s} test AUC {metrics['auc']:.3f} [{metrics['auc_ci_low']:.3f}, {metrics['auc_ci_high']:.3f}] "
              f"repeats {[round(a, 3) for a in rep_test_aucs]}  PR-AUC(fail) {metrics['pr_auc_failure']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Save test predictions
# MAGIC
# MAGIC **What:** writes every model's test-set probabilities to `gold.test_predictions`.
# MAGIC
# MAGIC **Why:** `06_compare` needs the *paired* predictions (same projects, different models) to test whether "combined beats structured" is a real gain or noise.

# COMMAND ----------

preds = pd.concat(test_preds, ignore_index=True)
spark.createDataFrame(preds).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.gold.test_predictions")

summary = {fs: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for fs, r in results.items()}
print(json.dumps(summary, indent=1))
dbutils.notebook.exit(json.dumps(summary))
