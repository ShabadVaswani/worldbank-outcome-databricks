# Databricks notebook source
# MAGIC %md
# MAGIC # 02_structured_silver — clean, leakage-safe structured data
# MAGIC
# MAGIC | Input (bronze) | Output (silver) | One row per |
# MAGIC |---|---|---|
# MAGIC | `ieg_ratings` | `silver.labels` | rated project |
# MAGIC | `projects` | `silver.projects` | project |
# MAGIC | `indicators` | `silver.country_year` | country × year |
# MAGIC
# MAGIC The three leakage rules applied here:
# MAGIC 1. **Labels table keeps only the label** — every IEG column measured at/after closing is dropped.
# MAGIC 2. **Project features are approval-time only** — closing date, status, effectiveness/ICR milestones and current commitments are dropped.
# MAGIC 3. **Country indicators are taken from the year before approval** (the join happens in `04_gold`; here we prepare the country-year table and only ever fill gaps *forward* in time).
# MAGIC
# MAGIC **Attach to:** `wb-ml-single`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings
# MAGIC
# MAGIC **What:** imports and table names.
# MAGIC
# MAGIC **Why:** the success definition lives here in one place: Moderately Satisfactory or better = 1.

# COMMAND ----------

import json
import numpy as np
import pandas as pd
from pyspark.sql import functions as F, Window

CATALOG = "dbw_wb_outcome"
SUCCESS = ["Highly Satisfactory", "Satisfactory", "Moderately Satisfactory"]
FAILURE = ["Moderately Unsatisfactory", "Unsatisfactory", "Highly Unsatisfactory"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Labels
# MAGIC
# MAGIC **What:** keeps rated projects only, makes `label` (1 = MS or better, 0 = MU or worse), removes the duplicate project ID by keeping its most recent evaluation, and keeps a few approval-time descriptors (instrument, agreement type, global practice).
# MAGIC
# MAGIC **Why:** `Not Rated` has no label. Duplicates would double-count a project. We deliberately **drop** `bank_performance`, `quality_at_entry`, `quality_of_supervision`, `me_quality`, `final_closing_fy`, and `country_fcs_*` — all decided after the fact. The one exception is `evaluation_fy`, kept as `rating_published_fy`: it is **not a feature**, it's the date the rating became public, which `04_gold` uses to make sure "track record" features only count ratings that existed before a project's approval.

# COMMAND ----------

ieg = spark.table(f"{CATALOG}.bronze.ieg_ratings").filter(F.col("outcome").isin(SUCCESS + FAILURE))
w = Window.partitionBy("project_id").orderBy(F.col("evaluation_fy").cast("int").desc_nulls_last())
labels = (ieg.withColumn("rn", F.row_number().over(w)).filter("rn = 1")
          .select("project_id",
                  F.col("outcome").alias("ieg_outcome"),
                  F.when(F.col("outcome").isin(SUCCESS), 1).otherwise(0).alias("label"),
                  F.col("approval_fy").cast("int").alias("ieg_approval_fy"),
                  F.col("evaluation_fy").cast("int").alias("rating_published_fy"),
                  "lending_instrument_type", "agreement_type", "global_practice"))
labels.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.silver.labels")

lab = spark.table(f"{CATALOG}.silver.labels")
print("labels:", lab.count(), "| distinct ids:", lab.select("project_id").distinct().count())
display(lab.groupBy("label").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Projects
# MAGIC
# MAGIC **What:** turns text amounts into numbers (US$ millions), parses the approval date, takes the first country code as the project's country (regional projects get `is_multi_country = 1`), extracts the first major sector, counts sectors and themes, and computes **preparation time** = months from concept review to approval.
# MAGIC
# MAGIC **Why:** XGBoost needs numbers. Preparation time is a classic signal (rushed or very slow preparation both correlate with trouble) and is known at approval. `closingdate`, `status`, `pdo`, effectiveness and ICR dates are **not** carried over.

# COMMAND ----------

def to_musd(col):
    return (F.regexp_replace(F.col(col).cast("string"), ",", "").cast("double") / 1e6)

p = spark.table(f"{CATALOG}.bronze.projects")
ms = F.col("milestones").getItem(0)

projects = (p.select(
        F.col("id").alias("project_id"),
        F.col("project_name"),
        F.to_date(F.substring("boardapprovaldate", 1, 10)).alias("approval_date"),
        F.col("fiscalyear").cast("int").alias("approval_fy"),
        F.col("regionname").alias("region"),
        F.col("countrycode").getItem(0).alias("country_iso2"),
        F.col("countryshortname").alias("country_name"),
        (F.size("countrycode") > 1).cast("int").alias("is_multi_country"),
        F.split("major_sector_name", ",").getItem(0).alias("major_sector"),
        F.size(F.split("sector_name", ",")).alias("n_sectors"),
        F.when(F.col("themecode").isNull(), 0).otherwise(F.size(F.split("themecode", ","))).alias("n_themes"),
        F.col("envassesmentcategorycode").alias("env_category"),
        to_musd("totalamt").alias("commitment_musd"),
        to_musd("grantamt").alias("grant_musd"),
        to_musd("idacommamt").alias("ida_musd"),
        to_musd("lendprojectcost").alias("project_cost_musd"),
        F.col("parentprojid").isNotNull().cast("int").alias("is_additional_financing"),
        F.to_date(F.substring(ms["concpt_review_date"], 1, 10)).alias("concept_review_date"),
        F.to_date(F.substring(ms["begin_apprsl_date"], 1, 10)).alias("appraisal_date"),
        F.col("status"))
    .withColumn("prep_months", F.months_between("approval_date", "concept_review_date"))
    .withColumn("appraisal_to_approval_months", F.months_between("approval_date", "appraisal_date"))
    .withColumn("prep_months", F.when(F.col("prep_months").between(0, 240), F.col("prep_months")))
    .withColumn("appraisal_to_approval_months",
                F.when(F.col("appraisal_to_approval_months").between(0, 120), F.col("appraisal_to_approval_months")))
    .withColumn("ida_share", F.when(F.col("commitment_musd") > 0, F.col("ida_musd") / F.col("commitment_musd")))
    .withColumn("cofinancing_ratio", F.when(F.col("commitment_musd") > 0, F.col("project_cost_musd") / F.col("commitment_musd")))
    .drop("concept_review_date", "appraisal_date"))

projects.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.silver.projects")
sp = spark.table(f"{CATALOG}.silver.projects")
print("projects:", sp.count())
display(sp.select([F.round(F.avg(F.col(c).isNotNull().cast("int")), 2).alias(c) for c in
                   ["approval_date", "country_iso2", "commitment_musd", "major_sector", "env_category", "prep_months"]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Country-year indicators
# MAGIC
# MAGIC **What:** pivots the long indicator table to one row per (country, year) with one column per indicator, then fills gaps **forward only** within each country, at most 3 years.
# MAGIC
# MAGIC **Why:** WGI was published every two years until 2002, and many series have occasional holes. Carrying the last known value forward is what an analyst at approval time would have had; filling *backward* would use future data, so we never do it. The 3-year cap stops very stale values from posing as current.

# COMMAND ----------

IND_NAMES = {
    "NY.GDP.PCAP.KD": "gdp_pc_const2015", "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "FP.CPI.TOTL.ZG": "inflation_pct", "DT.DOD.DECT.GN.ZS": "ext_debt_pct_gni",
    "SH.XPD.CHEX.GD.ZS": "health_exp_pct_gdp", "SE.XPD.TOTL.GD.ZS": "educ_exp_pct_gdp",
    "SP.POP.TOTL": "population", "NE.TRD.GNFS.ZS": "trade_pct_gdp",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_pct_gdp",
    "GOV_WGI_GE.EST": "wgi_gov_effectiveness", "GOV_WGI_RQ.EST": "wgi_regulatory_quality",
    "GOV_WGI_RL.EST": "wgi_rule_of_law", "GOV_WGI_CC.EST": "wgi_control_corruption",
    "GOV_WGI_PV.EST": "wgi_political_stability", "GOV_WGI_VA.EST": "wgi_voice_accountability",
}

ind = (spark.table(f"{CATALOG}.bronze.indicators")
       .filter((F.col("country_iso2") != "") & F.col("value").isNotNull())
       .select("country_iso2", F.col("year").cast("int").alias("year"), "indicator_id", "value")
       .toPandas())

wide = (ind.pivot_table(index=["country_iso2", "year"], columns="indicator_id", values="value", aggfunc="first")
        .rename(columns=IND_NAMES))
full_idx = pd.MultiIndex.from_product([wide.index.get_level_values(0).unique(), range(1980, 2026)],
                                      names=["country_iso2", "year"])
wide = wide.reindex(full_idx).sort_index()
wide = wide.groupby(level="country_iso2").ffill(limit=3).reset_index()
for c in IND_NAMES.values():
    if c not in wide.columns:
        wide[c] = float("nan")
wide["log_gdp_pc"] = np.log(wide["gdp_pc_const2015"].where(wide["gdp_pc_const2015"] > 0))
wide["log_population"] = np.log(wide["population"].where(wide["population"] > 0))
wide = wide.astype({c: "float64" for c in wide.columns if c not in ("country_iso2", "year")})

cy = spark.createDataFrame(wide.astype({"year": "int"}))
cy.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.silver.country_year")
print("country_year rows:", spark.table(f"{CATALOG}.silver.country_year").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Checks and summary
# MAGIC
# MAGIC **What:** confirms labels are unique per project and how many labeled projects can find their country in the indicator table.
# MAGIC
# MAGIC **Why:** a low country match would mean the ISO2 codes from the Projects API and the Indicators API disagree — better to catch it here than as mysteriously empty features in gold.

# COMMAND ----------

lab_proj = lab.join(sp, "project_id")
cy_keys = spark.table(f"{CATALOG}.silver.country_year").select("country_iso2").distinct()
country_hit = lab_proj.join(cy_keys, "country_iso2", "left_semi").count()

summary = {
    "labels": lab.count(),
    "label_rate": round(lab.agg(F.avg("label")).first()[0], 3),
    "labels_with_project": lab_proj.count(),
    "labels_with_country_indicators": country_hit,
    "country_year_rows": spark.table(f"{CATALOG}.silver.country_year").count(),
}
print(summary)
dbutils.notebook.exit(json.dumps(summary))
