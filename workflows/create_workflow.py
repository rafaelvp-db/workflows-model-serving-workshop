import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from dotenv import load_dotenv

load_dotenv() # Dev purposes

def main():

    w = WorkspaceClient(
        host = os.environ["DATABRICKS_HOST"],
        token = os.environ["DATABRICKS_TOKEN"]
    )

    notebook_path = f"""/Repos/{w.current_user.me().user_name}
        /workflows-model-serving-workshop/notebooks/ingest.py""".replace("\n", "")
    
    cluster_id = w.clusters.ensure_cluster_is_running(
        os.environ["DATABRICKS_CLUSTER_ID"]) and os.environ["DATABRICKS_CLUSTER_ID"]
    
    ingest_task = jobs.Task(
        description="test",
        existing_cluster_id=os.environ["DATABRICKS_CLUSTER_ID"],
        notebook_task=jobs.NotebookTask(notebook_path),
        task_key="test",
        timeout_seconds=0
    )

    created_job = w.jobs.create(
        name = f"sdk-{time.time_ns()}",
        tasks = [
            ingest_task
        ]
    )

    # cleanup
    w.jobs.delete(delete = "123")


if __name__ == "__main__":
    main()

