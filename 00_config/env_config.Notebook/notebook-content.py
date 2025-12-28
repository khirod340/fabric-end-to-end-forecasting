# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "035b2300-8442-49f8-b323-085c36a6b7b7",
# META       "default_lakehouse_name": "Sales_Lakehouse_DEV",
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

# =========================
# PARAMETER CELL
# =========================
# Default value for local testing.
# The Pipeline will inject a new cell BELOW this one to overwrite it.
ENV = "DEV"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==============================================================================
# CONFIGURATION & INFRASTRUCTURE LOGIC
# ==============================================================================
import sys

# 1. Validation: Ensure the pipeline actually passed the parameter
if 'ENV' not in locals() or ENV is None:
    raise RuntimeError("CRITICAL ERROR: 'ENV' parameter is missing. Pipeline injection failed.")

ENV = str(ENV).upper().strip()

# 2. Assign Variables based on Environment
if ENV == "DEV":
    LAKEHOUSE_NAME = "Sales_Lakehouse_DEV"
    MLFLOW_EXPERIMENT_NAME = "DEV_sales_forecasting"
    ALERT_EMAILS = ["khirod@tbf4v.onmicrosoft.com"]
    DRIFT_THRESHOLD = 0.25      # Loose threshold for dev
    ROLLBACK_THRESHOLD = 1.05   # Allow 5% degradation

elif ENV == "PROD":
    LAKEHOUSE_NAME = "Sales_Lakehouse_PROD"
    MLFLOW_EXPERIMENT_NAME = "PROD_sales_forecasting"
    ALERT_EMAILS = ["khirod@tbf4v.onmicrosoft.com", "manager@company.com"]
    DRIFT_THRESHOLD = 0.20      # Strict threshold for prod
    ROLLBACK_THRESHOLD = 1.02   # Allow only 2% degradation

else:
    raise ValueError(f"CRITICAL: Invalid ENV value '{ENV}'. Expected DEV or PROD.")

# 3. Log the Configuration
print("="*40)
print(f"✅ ENVIRONMENT ACTIVATED: {ENV}")
print(f"🎯 Target Lakehouse:      {LAKEHOUSE_NAME}")
print(f"🧪 MLflow Experiment:     {MLFLOW_EXPERIMENT_NAME}")
print("="*40)

# ==============================================================================
# SELF-HEALING INFRASTRUCTURE: FORCE SPARK CONTEXT
# ==============================================================================
# This ensures the notebook talks to the correct Lakehouse, 
# ignoring whatever is selected in the UI dropdown.

print(f"🔄 Infrastructure Check: Ensuring {LAKEHOUSE_NAME} exists...")

# 1. Create the Database/Lakehouse if it's missing (Self-Healing)
spark.sql(f"CREATE DATABASE IF NOT EXISTS {LAKEHOUSE_NAME}")

# 2. Switch Spark's active context to this Lakehouse
spark.sql(f"USE {LAKEHOUSE_NAME}")

print(f"🚀 SUCCESS: Spark is now actively querying '{LAKEHOUSE_NAME}'")

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
