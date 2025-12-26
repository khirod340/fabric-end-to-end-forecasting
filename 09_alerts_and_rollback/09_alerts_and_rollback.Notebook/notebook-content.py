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
import numpy as np
import mlflow
from pyspark.sql.functions import col, lit, current_timestamp

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

drift_df = spark.table("feature_drift_metrics")

latest_ts = drift_df.selectExpr("max(evaluation_timestamp)").collect()[0][0]

latest_drift_df = drift_df.filter(col("evaluation_timestamp") == latest_ts)

latest_drift_df.show(truncate=False)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

DRIFT_THRESHOLD = 0.25

BUSINESS_CRITICAL_FEATURES = ["is_promo", "discount_pct"]


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

alerts_schema = StructType([
    StructField("feature", StringType(), True),
    StructField("metric_value", DoubleType(), True),
    StructField("severity", StringType(), True),
    StructField("alert_type", StringType(), True),
    StructField("alert_timestamp", TimestampType(), True),
])

empty_alerts_df = spark.createDataFrame([], alerts_schema)

(
    empty_alerts_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("ml_alerts")
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

alerts_pd = []

for row in latest_drift_df.collect():
    feature = row["feature"]
    psi = row["psi"]
    
    if psi > DRIFT_THRESHOLD:
        severity = (
            "HIGH" if feature in BUSINESS_CRITICAL_FEATURES else "MEDIUM"
        )
        
        alerts_pd.append((
            feature,
            float(psi),
            severity,
            "DRIFT"
        ))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if alerts_pd:
    alerts_df = spark.createDataFrame(
        alerts_pd,
        ["feature", "metric_value", "severity", "alert_type"]
    ).withColumn(
        "alert_timestamp", current_timestamp()
    )

    (
        alerts_df
        .write
        .format("delta")
        .mode("append")
        .saveAsTable("ml_alerts")
    )
else:
    print("No alerts triggered.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

baseline = runs[runs["tags.model_role"] == "BASELINE"].iloc[0]
challenger = runs[runs["tags.model_role"] == "CHALLENGER"].iloc[0]

baseline_mae = baseline["metrics.MAE"]
challenger_mae = challenger["metrics.MAE"]

print("Baseline MAE:", baseline_mae)
print("Challenger MAE:", challenger_mae)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

ROLLBACK_THRESHOLD = 1.05  # 5% degradation allowed

severe_drift = any(
    (row["psi"] > DRIFT_THRESHOLD and row["feature"] in BUSINESS_CRITICAL_FEATURES)
    for row in latest_drift_df.collect()
)

rollback_required = (
    severe_drift and challenger_mae > baseline_mae * ROLLBACK_THRESHOLD
)

print("Severe drift detected:", severe_drift)
print("Rollback required:", rollback_required)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if rollback_required:
    rollback_df = spark.createDataFrame(
        [(
            "BASELINE",
            baseline["run_id"],
            "AUTO_ROLLBACK",
            pd.Timestamp.utcnow()
        )],
        ["active_model", "model_run_id", "reason", "decision_timestamp"]
    )

    (
        rollback_df
        .write
        .format("delta")
        .mode("overwrite")
        .saveAsTable("active_model_pointer")
    )

    print("Rollback executed: BASELINE model activated.")
else:
    print("No rollback executed. Monitoring continues.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.sql("SELECT * FROM ml_alerts").show(truncate=False)
spark.sql("SELECT * FROM active_model_pointer").show(truncate=False)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import current_timestamp

active_df = spark.table("active_model_pointer")

clean_active_df = (
    active_df
    .withColumn(
        "decision_timestamp",
        current_timestamp()
    )
)

(
    clean_active_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("active_model_pointer")
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.sql("SELECT * FROM active_model_pointer").show(truncate=False)


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
