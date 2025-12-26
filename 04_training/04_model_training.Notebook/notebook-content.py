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
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

gold_df = spark.table("gold_sales_features")

display(gold_df)


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

TARGET = "sales"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import max as spark_max, col, date_sub

max_date = gold_df.select(spark_max("date")).collect()[0][0]
cutoff_date = max_date - pd.Timedelta(days=90)

train_df = gold_df.filter(col("date") < cutoff_date)
test_df  = gold_df.filter(col("date") >= cutoff_date)

print("Train rows:", train_df.count())
print("Test rows:", test_df.count())


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

train_pd = train_df.select(FEATURES + [TARGET]).dropna().toPandas()
test_pd  = test_df.select(FEATURES + [TARGET]).dropna().toPandas()

X_train = train_pd[FEATURES]
y_train = train_pd[TARGET]

X_test = test_pd[FEATURES]
y_test = test_pd[TARGET]


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

model = LinearRegression()
model.fit(X_train, y_train)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run():
    
    # Params
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("features", ",".join(FEATURES))
    mlflow.log_param("train_window_days", "all_before_last_90_days")
    
    # Metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    
    # Tags
    mlflow.set_tag("env", ENV)
    mlflow.set_tag("model_role", "BASELINE")
    
    # Model artifact
    mlflow.sklearn.log_model(model, artifact_path="model")


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
