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

import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error
from pyspark.sql.functions import current_timestamp


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd
gold_df = spark.table("gold_sales_features")

max_date = gold_df.selectExpr("max(date)").collect()[0][0]
eval_start = max_date - pd.Timedelta(days=30)

eval_df = (
    gold_df
    .filter(gold_df.date >= eval_start)
    .select(
        "date",
        "store",
        "item",
        "sales",
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
    )
    .dropna()
)

eval_pd = eval_df.toPandas()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if eval_pd.empty:
    raise RuntimeError("No evaluation data available for performance monitoring.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

active_model_df = spark.table("active_model_pointer")

active_row = (
    active_model_df
    .orderBy("decision_timestamp", ascending=False)
    .limit(1)
    .collect()[0]
)

ACTIVE_MODEL = active_row["active_model"]
ACTIVE_RUN_ID = active_row["model_run_id"]

print("Active model:", ACTIVE_MODEL)
print("Active run_id:", ACTIVE_RUN_ID)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

active_model_df = spark.table("active_model_pointer")

active_row = (
    active_model_df
    .orderBy("decision_timestamp", ascending=False)
    .limit(1)
    .collect()[0]
)

ACTIVE_MODEL = active_row["active_model"]
ACTIVE_RUN_ID = active_row["model_run_id"]

print("Active model:", ACTIVE_MODEL)
print("Active run_id:", ACTIVE_RUN_ID)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

baseline_run = runs[runs["tags.model_role"] == "BASELINE"].iloc[0]
baseline_run_id = baseline_run.run_id


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

baseline_model = mlflow.sklearn.load_model(
    f"runs:/{baseline_run_id}/model"
)

active_model = mlflow.sklearn.load_model(
    f"runs:/{ACTIVE_RUN_ID}/model"
)


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

X_eval = eval_pd[FEATURES]
y_true = eval_pd["sales"]


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

baseline_pred = baseline_model.predict(X_eval)
active_pred = active_model.predict(X_eval)

baseline_mae = mean_absolute_error(y_true, baseline_pred)
active_mae = mean_absolute_error(y_true, active_pred)

degradation_pct = (active_mae - baseline_mae) / baseline_mae


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

DEGRADATION_THRESHOLD = ROLLBACK_THRESHOLD - 1  # e.g. 0.05

if degradation_pct > DEGRADATION_THRESHOLD:
    status = "DEGRADED"
else:
    status = "OK"

print("Baseline MAE:", baseline_mae)
print("Active MAE:", active_mae)
print("Degradation %:", degradation_pct)
print("STATUS:", status)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print("baseline_mae:", baseline_mae, type(baseline_mae))
print("active_mae:", active_mae, type(active_mae))
print("degradation_pct:", degradation_pct, type(degradation_pct))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

baseline_mae = float(baseline_mae)
active_mae = float(active_mae)
degradation_pct = float(degradation_pct)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

performance_schema = StructType([
    StructField("active_model", StringType(), False),
    StructField("baseline_mae", DoubleType(), False),
    StructField("active_model_mae", DoubleType(), False),
    StructField("degradation_pct", DoubleType(), False),
    StructField("status", StringType(), False),
    StructField("evaluation_timestamp", TimestampType(), False)
])


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from datetime import datetime

row_data = [(
    ACTIVE_MODEL,
    baseline_mae,
    active_mae,
    degradation_pct,
    status,
    datetime.utcnow()
)]

result_df = spark.createDataFrame(
    row_data,
    schema=performance_schema
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

(
    result_df
    .write
    .format("delta")
    .mode("append")
    .saveAsTable("model_performance_metrics")
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# MAGIC %%sql
# MAGIC SELECT *
# MAGIC FROM model_performance_metrics
# MAGIC ORDER BY evaluation_timestamp DESC;


# METADATA ********************

# META {
# META   "language": "sparksql",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
