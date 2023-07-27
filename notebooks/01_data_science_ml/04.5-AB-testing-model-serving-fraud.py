# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # New Model deployement with A/B testing 
# MAGIC
# MAGIC
# MAGIC <img style="float: right;" src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/fsi/fraud-detection/model-serving-ab-testing.png" width="800px" />
# MAGIC
# MAGIC Our new model is now saved in our Registry.
# MAGIC
# MAGIC Our next step is now to deploy it while ensuring that it's behaving as expected. We want to be able to deploy the new version in the REST API:
# MAGIC
# MAGIC * Without making any production outage
# MAGIC * Slowly routing requests to the new model
# MAGIC * Supporting auto-scaling & potential bursts
# MAGIC * Performing some A/B testing ensuring the new model is providing better outcomes
# MAGIC * Monitorig our model outcome and technical metrics (CPU/load etc)
# MAGIC
# MAGIC Databricks makes this process super simple with Serverless Model Serving endpoint.
# MAGIC
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Ffsi%2Flakehouse_fsi_fraud%2Fml-ab-testing&dt=LAKEHOUSE_FSI_FRAUD">

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Routing our Model Serving endpoint to multiple models
# MAGIC <img style="float: right; margin-left: 10px" width="700px" src="https://cms.databricks.com/sites/default/files/inline-images/db-498-blog-imgs-1.png" />
# MAGIC
# MAGIC Databricks Model Serving endpoints allow you to serve different models and dynamically redirect a subset of the traffic to a given model.
# MAGIC
# MAGIC Open your <a href="#mlflow/endpoints/dbdemos_fsi_fraud" target="_blank"> Model Serving Endpoint</a>, edit the configuration and add our second model.
# MAGIC
# MAGIC Select the traffic ratio you want to send to the new model (20%), save and Databricks will handle the rest for you. 
# MAGIC
# MAGIC Your endpoint will automatically bundle the new model, and start routing a subset of your queries to this model.
# MAGIC
# MAGIC Let's see how this can be done using the API.

# COMMAND ----------

latest_model

# COMMAND ----------

client.get_latest_versions(model_name)[0]

# COMMAND ----------

# DBTITLE 1,Move the new model in production if it's not already the case
import mlflow

model_name = "demo_fraud_rvp"

client = mlflow.tracking.MlflowClient()
latest_model = client.get_latest_versions(model_name)[0]
if latest_model.current_stage != 'Production':
  client.transition_model_version_stage(model_name, latest_model.version, stage = "Production", archive_existing_versions=False)

# COMMAND ----------

# MAGIC %pip install git+https://github.com/sebrahimi1988/databricks-model-serving

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

latest_model.version

# COMMAND ----------

# DBTITLE 1,Deploying our model with a new route using the API
from databricks.model_serving.client import EndpointClient

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

serving_client = EndpointClient(databricks_url, databricks_token)

model_currently_served = serving_client.get_inference_endpoint(model_name)['config']['served_models'][0]

models = [
  { 
    "name": f"{model_name}-A",
    "model_name": model_name,
    "model_version": latest_model.version,
    "workload_size": "Small",
    "scale_to_zero_enabled": True
  }, 
  {
    "name": f"{model_name}-B",
    "model_name": model_name,
    "model_version": model_currently_served['model_version'],
    "workload_size": "Small",
    "scale_to_zero_enabled": True
  }
]

traffic_config = {
  "routes": [
    {
      "served_model_name": f"{model_name}-A",
      "traffic_percentage": 20
    },
    {
      "served_model_name": f"{model_name}-B",
      "traffic_percentage": 80
    }
  ]
}

serving_client.update_served_models(model_name, models, traffic_config)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Our new model is now serving 20% of our requests
# MAGIC
# MAGIC Open your <a href="#mlflow/endpoints/demo_fraud_rvp" target="_blank"> Model Serving Endpoint</a> to view the changes and track the 2 models performance

# COMMAND ----------

# DBTITLE 1,Trying our new Model Serving setup
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

dataset = spark.sql("select * from dbdemos.fsi_fraud_detection.transactions_features limit 30").toPandas()

payload = {
  "dataframe_records": dataset.to_dict(orient = "records")
}

endpoint_url = f"{serving_client.base_url}/realtime-inference/{model_name}/invocations"
inferences = serving_client.query_inference_endpoint(model_name, data = payload)
print("Fraud inference:")
print(inferences)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Model monitoring and A/B testing analysis
# MAGIC
# MAGIC Because the Model Serving runs within our Lakehouse, Databricks will automatically save and track all our Model Endpoint results as a Delta Table.
# MAGIC
# MAGIC We can then easily plug a feedback loop to start analysing the revenue in $ each model is offering. 
# MAGIC
# MAGIC All these metrics, including A/B testing validation (p-values etc) can then be pluged into a Model Monitoring Dashboard and alerts can be sent for errors, potentially triggering new model retraining or programatically updating the Endpoint routes to fallback to another model.
# MAGIC
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/fsi/fraud-detection/model-serving-monitoring.png" width="1200px" />

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Conclusion: the power of the Lakehouse
# MAGIC
# MAGIC In this demo, we've seen an end 2 end flow with the Lakehouse:
# MAGIC
# MAGIC - Data ingestion made simple with Delta Live Table
# MAGIC - Leveraging Databricks warehouse to Analyze existing Fraud
# MAGIC - Model Training with AutoML for citizen Data Scientist
# MAGIC - Ability to tune our model for better results, improving our revenue
# MAGIC - Ultimately, the ability to Deploy and track our models in real time, made possible with the full lakehouse capabilities.
# MAGIC
# MAGIC [Go back to the introduction]($../00-FSI-fraud-detection-introduction-lakehouse) or discover how to use Databricks Workflow to orchestrate this tasks: [05-Workflow-orchestration-fsi-fraud]($../05-Workflow-orchestration/05-Workflow-orchestration-fsi-fraud)
