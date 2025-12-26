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

silver_df = spark.table("silver_sales")

display(silver_df)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import (
    col,
    dayofweek,
    weekofyear,
    month
)

calendar_df = (
    silver_df
    .withColumn("day_of_week", dayofweek(col("date")))
    .withColumn("week_of_year", weekofyear(col("date")))
    .withColumn("month", month(col("date")))
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.window import Window
from pyspark.sql.functions import lag, avg

window_spec = (
    Window
    .partitionBy("store", "item")
    .orderBy("date")
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

features_df = (
    calendar_df
    .withColumn("sales_lag_7", lag(col("sales"), 7).over(window_spec))
    .withColumn("sales_lag_14", lag(col("sales"), 14).over(window_spec))
    .withColumn("sales_lag_28", lag(col("sales"), 28).over(window_spec))
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import avg

roll_window_7 = window_spec.rowsBetween(-7, -1)
roll_window_28 = window_spec.rowsBetween(-28, -1)

features_df = (
    features_df
    .withColumn("sales_roll_mean_7", avg(col("sales")).over(roll_window_7))
    .withColumn("sales_roll_mean_28", avg(col("sales")).over(roll_window_28))
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

gold_df = features_df.select(
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

display(gold_df)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

(
    gold_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("gold_sales_features")
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.sql("SELECT COUNT(*) AS gold_rows FROM gold_sales_features").show()

spark.sql("""
SELECT
  COUNT(*) AS rows_with_null_lag
FROM gold_sales_features
WHERE sales_lag_7 IS NULL
""").show()

spark.sql("""
SELECT
  MIN(date) AS min_date,
  MAX(date) AS max_date
FROM gold_sales_features
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
