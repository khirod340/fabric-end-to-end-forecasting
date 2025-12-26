# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
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

# =========================
# CONFIGURATION LOGIC
# =========================

# 1. Validation
if ENV is None:
    raise RuntimeError(
        "ENV parameter is missing. Pipeline injection failed."
    )

ENV = str(ENV).upper().strip()

# 2. Assign Variables
if ENV == "DEV":
    LAKEHOUSE_NAME = "Sales_Lakehouse_DEV"
    MLFLOW_EXPERIMENT_NAME = "DEV_sales_forecasting"
    ALERT_EMAILS = ["dev-ml@company.com"]
    DRIFT_THRESHOLD = 0.25
    ROLLBACK_THRESHOLD = 1.05

elif ENV == "PROD":
    LAKEHOUSE_NAME = "Sales_Lakehouse_PROD"
    MLFLOW_EXPERIMENT_NAME = "PROD_sales_forecasting"
    ALERT_EMAILS = ["ml-ops@company.com"]
    DRIFT_THRESHOLD = 0.20
    ROLLBACK_THRESHOLD = 1.02

else:
    raise ValueError(
        f"CRITICAL: Invalid ENV '{ENV}'. Expected DEV or PROD."
    )

print(f"✅ CONFIG LOADED: {ENV}")
print(f"Lakehouse: {LAKEHOUSE_NAME}")


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
