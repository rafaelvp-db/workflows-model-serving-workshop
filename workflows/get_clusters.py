from databricks.sdk import WorkspaceClient
import os

w = WorkspaceClient(
    host = os.environ["DATABRICKS_HOST"],
    token = os.environ["DATABRICKS_TOKEN"]
)

for c in w.clusters.list():
  print(c.cluster_id)
  if c.cluster_id == "0727-054928-41plse7n":
    print("### Exists!")