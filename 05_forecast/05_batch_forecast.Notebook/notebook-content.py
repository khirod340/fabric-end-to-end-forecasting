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
from pyspark.sql.functions import current_timestamp, lit

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

latest_run = runs.sort_values("start_time", ascending=False).iloc[0]
run_id = latest_run.run_id

model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

print("Using run_id:", run_id)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

gold_df = spark.table("gold_sales_features")

max_date = gold_df.selectExpr("max(date)").collect()[0][0]

predict_df = gold_df.filter(gold_df.date == max_date)


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

predict_pd = (
    predict_df
    .select("date", "store", "item", *FEATURES)
    .dropna()
    .toPandas()
)

X_pred = predict_pd[FEATURES]


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

predict_pd["predicted_sales"] = model.predict(X_pred)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

predict_pd["model_run_id"] = run_id
predict_pd["model_role"] = "BASELINE"
predict_pd["prediction_timestamp"] = pd.Timestamp.utcnow()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

forecast_spark_df = spark.createDataFrame(predict_pd)

(
    forecast_spark_df
    .write
    .format("delta")
    .mode("append")
    .saveAsTable("gold_sales_forecast")
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.sql("""
SELECT
  COUNT(*) AS forecast_rows,
  COUNT(DISTINCT model_run_id) AS model_versions
FROM gold_sales_forecast
""").show()

spark.sql("""
SELECT
  MIN(prediction_timestamp),
  MAX(prediction_timestamp)
FROM gold_sales_forecast
""").show()


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
