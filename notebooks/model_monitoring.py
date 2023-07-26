# Databricks notebook source
# MAGIC %md
# MAGIC # Data Monitoring Quickstart for ML Inference-like table
# MAGIC
# MAGIC **System requirements:**
# MAGIC - ML runtime [11.3LTS+](https://docs.databricks.com/release-notes/runtime/11.3ml.html)
# MAGIC - [Unity-Catalog enabled workspace](https://docs.databricks.com/data-governance/unity-catalog/enable-workspaces.html)
# MAGIC - Disabled **Customer-Managed-Key(s)** for encryption [AWS](https://docs.databricks.com/security/keys/customer-managed-keys-managed-services-aws.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/security/keys/customer-managed-key-managed-services-azure)|[GCP]()
# MAGIC
# MAGIC [Link](https://drive.google.com/drive/u/0/folders/1oXuP-VleXmq0fTE4YovavboAVC7L-DF5) to google drive containing:
# MAGIC - User guide on core concepts
# MAGIC - API reference for API details and guidelines 
# MAGIC
# MAGIC In this notebook we'll monitor a _(batch)_ **Inference ML table** for an ML _regression_ model

# COMMAND ----------

# MAGIC %pip install "https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_data_monitoring-0.1.0-py3-none-any.whl"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import databricks.data_monitoring as dm
from databricks.data_monitoring import analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pre-Requisites
# MAGIC * an **existing DELTA table in Unity-Catalog created/owned by current_user** _(running the notebook)_, containing **batch scoring** or **inference logs** data with the following **mandatory columns**
# MAGIC   * `timestamp` column _(TimeStamp)_ (used for windowing/aggregation when calculatin drift/metrics)
# MAGIC   * `model_version` column _(String)_ model version used for each infered/scored observation
# MAGIC   * `prediction` column _(String)_ containing model prediction outputs
# MAGIC   * _(OPTIONAL)_ `label` column _(String)_ containing ground-truth data
# MAGIC   * _(RECOMMENDED)_ enable Delta's [Change-Data-Feed](https://docs.databricks.com/delta/delta-change-data-feed.html#enable-change-data-feed) table property on monitored (and baseline) table(s) for better performance
# MAGIC * _(OPTIONAL)_ an existing **baseline (DELTA) table** containing same data/column names as monitored table in addition to `model_version` column with Change-Data-Feed property enabled as well
# MAGIC * _(OPTIONAL)_ an **existing** _(dummy)_ **model in MLflow's model registry** (under `models:/registry_model_name`, in order to visualize monitoring UI and links to DBSQL dashboard)

# COMMAND ----------

# MAGIC %md
# MAGIC To enable Change-Data-Feed, couple of options:
# MAGIC 1. At creation time (SQL: `TBLPROPERTIES (delta.enableChangeDataFeed = true)`/pyspark: `.option("delta.enableChangeDataFeed", "true")`)
# MAGIC 2. Ad-hoc (SQL: `ALTER TABLE myDeltaTable SET TBLPROPERTIES (delta.enableChangeDataFeed = true)`)
# MAGIC 3. Bulk set for every created delta table as part of a notebook: `set spark.databricks.delta.properties.defaults.enableChangeDataFeed = true;`

# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply("user")
username_prefix = username.split("@")[0].replace(".","_")

dbutils.widgets.text("table_name", f"{username_prefix}_airbnb_bookings_model", "Table to Monitor")
dbutils.widgets.text("baseline_table_name",f"{username_prefix}_airbnb_bookings_baseline", "Baseline table (OPTIONAL)")
dbutils.widgets.text("monitor_db", f"{username_prefix}_monitor_db", "Output Database/Schema to use (OPTIONAL)")
dbutils.widgets.text("monitor_catalog", "uc_demos_rvp", "Unity Catalog to use (Required)")
dbutils.widgets.text("problem_type", "regression", "Type of Problem (regression/classification)")

# COMMAND ----------

# Required parameters in order to run this notebook.
CATALOG = dbutils.widgets.get("monitor_catalog")
TABLE_NAME = dbutils.widgets.get("table_name")
QUICKSTART_MONITOR_DB = dbutils.widgets.get("monitor_db") # Output database/schema to store analysis/drift metrics tables in
BASELINE_TABLE = dbutils.widgets.get("baseline_table_name")  # OPTIONAL - Baseline table name, if any, for computing drift against baseline

print(CATALOG, TABLE_NAME, QUICKSTART_MONITOR_DB, BASELINE_TABLE)

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG $monitor_catalog;
# MAGIC CREATE SCHEMA IF NOT EXISTS $monitor_db;
# MAGIC USE $monitor_db;
# MAGIC DROP TABLE IF EXISTS $table_name;

# COMMAND ----------

try:
  # Using API call
  dm.delete_monitor(
      table_name=f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}",
      purge_artifacts=True,
  )

except dm.errors.DataMonitoringError as e:
  if e.error_code == "MONITOR_NOT_FOUND":
      print(
          f"No existing monitor on table {CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}!"
      )

  elif e.error_code == "DATA_MONITORING_SERVICE_ERROR":
      print(e.message)

  else:
      raise (e)

# COMMAND ----------

display(spark.read.table(f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{BASELINE_TABLE}"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1.1 Train Regression Model

# COMMAND ----------

from sklearn.model_selection import train_test_split
import numpy as np

df = spark.read.table(f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{BASELINE_TABLE}").toPandas()

cat_vars = [
  "host_is_superhost",
  "cancellation_policy",
  "instant_bookable",
  "neighbourhood_cleansed",
  "property_type",
  "room_type",
  "bed_type"
]

df = df.drop(cat_vars, axis = 1)

df_val = df.sample(frac = 0.10)

df_train_test = df.iloc[[index for index in df.index if not np.isin(index, df_val.index)]]

X_train, X_test, y_train, y_test = train_test_split(
  df_train_test.drop("price", axis=1),
  df_train_test.price.values,
  test_size = 0.3
)

# COMMAND ----------

from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import mlflow

with mlflow.start_run() as run:

  model = LGBMRegressor()
  model.fit(X_train, y_train)
  yhat_train = model.predict(X_train)
  r2_train = r2_score(y_train, yhat_train)

  mlflow.log_metric("r2_train", r2_train)

  yhat_test = model.predict(X_test)
  r2_test = r2_score(y_test, yhat_test)
  mlflow.log_metric("r2_test", r2_test)

  mlflow.sklearn.log_model(
    sk_model = model,
    artifact_path="model",
    registered_model_name = "airbnb_lgbm"
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. Predict (batch inference) on our incoming data test data 
# MAGIC
# MAGIC Day 1

# COMMAND ----------

import mlflow
import sklearn
from datetime import timedelta, datetime
from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "airbnb_lgbm"
model_stage = "None"
version = client.get_latest_versions(name=model_name, stages=[model_stage])[0].version
new_loaded_model = mlflow.pyfunc.spark_udf(
  spark,
  model_uri=f"models:/{model_name}/{version}",
  result_type="double"
)

# COMMAND ----------

import pyspark.sql.functions as F

TIMESTAMP_COL = "date_time"
features_list = X_test.columns

# Add/Simulate timestamp(s) if they don't exist
timestamp = (datetime.now()).timestamp()
testDF = spark.createDataFrame(X_test)
predDF = testDF.withColumn(
  TIMESTAMP_COL,
  F.lit(timestamp).cast("timestamp")
  ).withColumn("prediction", new_loaded_model(*features_list))

display(predDF)

# COMMAND ----------

MODEL_VERSION_COL = "model_version"

predDF.withColumn(
  MODEL_VERSION_COL, F.lit(version)) \
    .write.format("delta").mode("overwrite") \
    .option("overwriteSchema",True) \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Monitor
# MAGIC Using `InferenceLog` type analysis
# MAGIC
# MAGIC **Required parameters**:
# MAGIC - `TABLE_NAME`: Name of the table to monitor.
# MAGIC - `PROBLEM_TYPE`: ML problem type, for `problem_type` parameter of `monitor_table`. Either `"classification"` or `"regression"`.
# MAGIC - `PREDICTION_COL`: Name of column in `TABLE_NAME` storing model predictions.
# MAGIC - `TIMESTAMP_COL`: Name of `timestamp` column in inference table
# MAGIC - `MODEL_VERSION_COL`: Name of column reflecting model version.
# MAGIC - `GRANULARITIES`: Monitoring analysis granularities (i.e. `["5 minutes", "30 minutes", "1 hour", "1 day", "n weeks", "1 month", "1 year"]`)
# MAGIC
# MAGIC **Optional parameters**:
# MAGIC - `OUTPUT_SCHEMA_NAME`: _(OPTIONAL)_ Name of the database/schema where to create output tables (can be either {schema} or {catalog}.{schema} format). If not provided default/current DB will be used.
# MAGIC - `LINKED_ENTITIES` _(OPTIONAL but useful for Private Preview in order to visualize monitoring UI)_: List of Databricks entity names that are associated with this table. **Only following entities are supported:**
# MAGIC      - `["models:/registry_model_name", "models:/my_model"]` links model(s) in the MLflow registry to the monitored table.
# MAGIC
# MAGIC **Monitoring parameters**:
# MAGIC - `BASELINE_TABLE_NAME` _(OPTIONAL)_: Name of table containing baseline data **NEEDS TO HAVE A `model_version` COLUMN** in case of `InferenceLog` analysis
# MAGIC - `SLICING_EXPRS` _(OPTIONAL)_: List of column expressions to independently slice/group data for analysis. (i.e. `slicing_exprs=["col_1", "col_2 > 10"]`)
# MAGIC - `CUSTOM_METRICS` _(OPTIONAL)_: A list of custom metrics to compute alongside existing aggregate, derived, and drift metrics.
# MAGIC - `SKIP_ANALYSIS` _(OPTIONAL)_: Flag to run analysis at monitor creation/update invoke time.
# MAGIC - `DATA_MONITORING_DIR` _(OPTIONAL)_: absolute path to existing directory for storing artifacts under `/{table_name}` (default=`/Users/{user_name}/databricks_data_monitoring`)
# MAGIC
# MAGIC **Table parameters** :
# MAGIC - `LABEL_COL` _(OPTIONAL)_: Name of column storing labels
# MAGIC - `EXAMPLE_ID_COL` _(OPTIONAL)_: Name of (unique) identifier column (to be ignored in analysis)
# MAGIC
# MAGIC **Make sure to drop any column that you don't want to track or which doesn't make sense from a business or use-case perspective**

# COMMAND ----------

problem_type = dbutils.widgets.get("problem_type")  # ML problem type, one of "classification"/"regression"

# Validate that all required inputs have been provided
if None in [model_name, PROBLEM_TYPE]:
    raise Exception("Please fill in the required information for model name and problem type.")

# Monitoring configuration parameters, see create_or_update_monitor() documentation for more details.
GRANULARITIES = ["1 day"]                       # Window sizes to analyze data over

# Optional parameters to control monitoring analysis.
LABEL_COL = "price"                             # Name of columns holding labels
SLICING_EXPRS = ["cancellation_policy", "accommodates > 2"]   # Expressions to slice data with
LINKED_ENTITIES = [f"models:/{model_name}"]
# DATA_MONITORING_DIR = f"/Users/{username}/DataMonitoringTEST"

# Parameters to control processed tables.
PREDICTION_COL = "prediction"  # What to name predictions in the generated tables
EXAMPLE_ID_COL = "id" # Optional

# Custom Metrics
CUSTOM_METRICS = None # for now

# COMMAND ----------

spark.sql("select * from rafael_pierre_airbnb_bookings_model").printSchema()

# COMMAND ----------

print(f"Creating monitor for {TABLE_NAME}")

dm_info = dm.create_or_update_monitor(
    table_name=f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{BASELINE_TABLE}", 
    granularities=GRANULARITIES,
    analysis_type=analysis.InferenceLog(
        timestamp_col="date_time",
        example_id_col="id", # To drop from analysis
        model_version_col="model_version", # Version number
        prediction_col="prediction",
        label_col="price",
        problem_type=problem_type,
    ),
    output_schema_name=QUICKSTART_MONITOR_DB,
    baseline_table_name=BASELINE_TABLE,
    slicing_exprs=SLICING_EXPRS,
    linked_entities=LINKED_ENTITIES
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3.1 Inspect the analysis tables 
# MAGIC
# MAGIC Notice that the cell below shows that within the monitor_db, there are four other tables:
# MAGIC
# MAGIC 1. analysis_metrics
# MAGIC 2. drift_metrics
# MAGIC
# MAGIC These two tables (analysis_metrics and drift_metrics) record the outputs of analysis jobs.

# COMMAND ----------

# MAGIC %sql 
# MAGIC SHOW TABLES
# MAGIC -- QUICKSTART_MONITOR_DB name or a widgest monitor_db
# MAGIC FROM default LIKE '$table_name*'

# COMMAND ----------

# DBTITLE 1,Analysis_Metrics Table
analysisDF = spark.sql(f"SELECT * FROM {dm_info.assets.analysis_metrics_table_name}")
display(analysisDF)

# COMMAND ----------

# MAGIC %md
# MAGIC You can see that for every column, the analysis table differentiates baseline data from scoring data and generates analyses based on:
# MAGIC - window
# MAGIC - model version
# MAGIC - slice key
# MAGIC
# MAGIC We can also gain insight into basic summary statistics
# MAGIC - percent_distinct
# MAGIC - data_type
# MAGIC - min
# MAGIC - max
# MAGIC - etc.

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the drift table below, we are able to tell the shifts / changes between the `trainDF` and `testDF1`. 

# COMMAND ----------

display(spark.sql(f"SELECT column_name, * FROM {dm_info.assets.drift_metrics_table_name}"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Since this comparison of `testDF1` is made against the baseline `trainDF`, we can see that `drift_type = "BASELINE"`. We will see another drift type, called `"CONSECUTIVE"`, when we have multiple batches of `scored_data` to compare.

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {dm_info.assets.drift_metrics_table_name}").groupby("drift_type").count())

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Adding some customer Metrics 

# COMMAND ----------

from pyspark.sql import types as T
from databricks.data_monitoring.metrics import Metric

CUSTOM_METRICS = [
    Metric(
           metric_type="aggregate",
           metric_name="log_avg",
           input_columns=["price"],
           metric_definition="avg(log(abs(`{{column_name}}`)+1))",
           output_type=T.DoubleType()
           ),
    Metric(
           metric_type="derived",
           metric_name="exp_log",
           input_columns=["price"],
           metric_definition="exp(log_avg)",
           output_type=T.DoubleType()
        ),
    Metric(
           metric_type="drift",
           metric_name="delta_exp",
           input_columns=["price"],
           metric_definition="{{current_df}}.exp_log - {{base_df}}.exp_log",
           output_type=T.DoubleType()
        )
]

# COMMAND ----------

dm.update_monitor(table_name=f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}",
                  updated_params={
                   "custom_metrics" : CUSTOM_METRICS
                 })

# COMMAND ----------

dm.refresh_metrics(table_name=f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}",
                   backfill=True)

# COMMAND ----------


