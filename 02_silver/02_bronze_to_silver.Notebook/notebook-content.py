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

%run "env_config"



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

bronze_df = spark.table("bronze_sales")

display(bronze_df)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import (
    col,
    when,
    lit,
    row_number,
    current_date
)
from pyspark.sql.window import Window

# -------------------------
# 1. Basic field cleaning
# -------------------------

clean_df = (
    bronze_df
    # Date must exist and must not be far future
    .filter(col("date").isNotNull())
    .filter(col("date") <= current_date() + lit(1))
    
    # Sales: NULL or negative -> 0
    .withColumn(
        "sales",
        when(col("sales").isNull() | (col("sales") < 0), lit(0))
        .otherwise(col("sales"))
    )
    
    # Promotions: NULL -> 0
    .withColumn(
        "is_promo",
        when(col("is_promo").isNull(), lit(0)).otherwise(col("is_promo"))
    )
    .withColumn(
        "discount_pct",
        when(col("discount_pct").isNull(), lit(0.0))
        .otherwise(col("discount_pct"))
    )
)

# -------------------------
# 2. Drop invalid business keys
# -------------------------

clean_df = clean_df.filter(
    col("store").isNotNull() & col("item").isNotNull()
)

# -------------------------
# 3. Deduplication
# -------------------------

window_spec = Window.partitionBy(
    "date", "store", "item"
).orderBy(col("ingestion_timestamp").desc())

silver_df = (
    clean_df
    .withColumn("row_num", row_number().over(window_spec))
    .filter(col("row_num") == 1)
    .drop("row_num")
)

display(silver_df)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

(
    silver_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("silver_sales")
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.sql("SELECT COUNT(*) AS silver_rows FROM silver_sales").show()

spark.sql("""
SELECT
  SUM(is_promo) AS promo_rows,
  SUM(sales) AS total_sales
FROM silver_sales
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
