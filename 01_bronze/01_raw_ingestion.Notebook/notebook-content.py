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

raw_path = "Files/raw/train.csv"

raw_df = (
    spark.read
    .option("header", True)
    .csv(raw_path)
)

display(raw_df)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col

typed_df = (
    raw_df
    .withColumn("date", col("date").cast("date"))
    .withColumn("sales", col("sales").cast("int"))
)

display(typed_df)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import date_add, col

# Shift all dates forward by ~6 years
DATE_OFFSET_DAYS = 365 * 6

shifted_df = (
    typed_df
    .withColumn(
        "date",
        date_add(col("date"), DATE_OFFSET_DAYS)
    )
)

display(shifted_df)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

shifted_df.selectExpr(
    "min(date) as min_date",
    "max(date) as max_date"
).show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import current_timestamp

bronze_df = (
    shifted_df
    .withColumn(
        "ingestion_timestamp",
        current_timestamp()
    )
)

display(bronze_df)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

(
    bronze_df
    .write
    .format("delta")
    .mode("append")
    .saveAsTable("bronze_sales")
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.sql("SELECT COUNT(*) AS row_count FROM bronze_sales").show()

spark.sql("""
SELECT 
  MIN(date) AS min_date,
  MAX(date) AS max_date
FROM bronze_sales
""").show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, abs, hash, when, lit

# Deterministic promo assignment using business keys
promo_fixed_df = (
    bronze_df
    .withColumn(
        "_promo_hash",
        abs(hash(col("store"), col("item"), col("date"))) % 100
    )
    .withColumn(
        "is_promo",
        when(col("_promo_hash") < 10, lit(1)).otherwise(lit(0))
    )
    .withColumn(
        "discount_pct",
        when(
            col("is_promo") == 1,
            (col("_promo_hash") / 100.0) * 0.30
        ).otherwise(lit(0.0))
    )
    .drop("_promo_hash")
)

display(promo_fixed_df)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

(
    promo_fixed_df
    .write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable("bronze_sales")
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.sql("DESCRIBE TABLE bronze_sales").show(truncate=False)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.sql("""
SELECT
  COUNT(*) AS total_rows,
  SUM(is_promo) AS promo_rows
FROM bronze_sales
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
