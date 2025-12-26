# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "035b2300-8442-49f8-b323-085c36a6b7b7",
# META       "default_lakehouse_name": "Sales_Lakehouse",
# META       "default_lakehouse_workspace_id": "ed951dd5-31a7-4bea-abcd-8d57aaa6b4e0",
# META       "known_lakehouses": [
# META         {
# META           "id": "035b2300-8442-49f8-b323-085c36a6b7b7"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

%run "env_config"



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

gold_df = spark.table("gold_sales_features")

max_date = gold_df.selectExpr("max(date)").collect()[0][0]
eval_start = max_date - pd.Timedelta(days=30)

actuals_df = gold_df.filter(gold_df.date >= eval_start)

actuals_pd = (
    actuals_df
    .select("date", "store", "item", "sales")
    .toPandas()
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

baseline_run = runs[runs["tags.model_role"] == "BASELINE"].iloc[0]
challenger_run = runs[runs["tags.model_role"] == "CHALLENGER"].iloc[0]

baseline_model = mlflow.sklearn.load_model(f"runs:/{baseline_run.run_id}/model")
challenger_model = mlflow.sklearn.load_model(f"runs:/{challenger_run.run_id}/model")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

FEATURES = [
    "is_promo",
    "discount_pct",
    "day_of_week",
    "week_of_year",
    "month",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_roll_mean_7",
    "sales_roll_mean_28"
]

eval_pd = (
    actuals_df
    .select("date", "store", "item", "sales", *FEATURES)
    .dropna()
    .toPandas()
)

X_eval = eval_pd[FEATURES]
y_true = eval_pd["sales"]


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

eval_pd["pred_baseline"] = baseline_model.predict(X_eval)
eval_pd["pred_challenger"] = challenger_model.predict(X_eval)

baseline_mae = mean_absolute_error(y_true, eval_pd["pred_baseline"])
challenger_mae = mean_absolute_error(y_true, eval_pd["pred_challenger"])

print("Baseline MAE:", baseline_mae)
print("Challenger MAE:", challenger_mae)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Challenger must beat baseline by at least 5%
IMPROVEMENT_THRESHOLD = 0.05

if challenger_mae < baseline_mae * (1 - IMPROVEMENT_THRESHOLD):
    active_model = "CHALLENGER"
    active_run_id = challenger_run.run_id
else:
    active_model = "BASELINE"
    active_run_id = baseline_run.run_id

print("ACTIVE MODEL:", active_model)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

active_model_df = spark.createDataFrame(
    [(active_model, active_run_id, pd.Timestamp.utcnow())],
    ["active_model", "model_run_id", "decision_timestamp"]
)

(
    active_model_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("active_model_pointer")
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
