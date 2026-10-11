"""
Promote the latest version of the registered model to Production
if the weighted F1 score from the most recent evaluation run exceeds
the required threshold.

Run:
    python scripts/promote.py
"""
# TODO: Write a script that :
# 1- reads the F1 threshold from params.yaml,
# 2- fetches the weighted F1 score from the most recent MLflow evaluation run,
# 3- promotes the latest model version to a "production" alias if the score
# meets the threshold.
# 4- Print a clear message when the model does not qualify.
# Documentation: https://mlflow.org/docs/latest/model-registry.html

import os

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

PRODUCTION_ALIAS = "production"


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["promote"]


def get_latest_f1(client: MlflowClient, experiment_name: str) -> tuple[str, float]:
    """
    Return the run_id and f1_weighted of the most recent evaluation run.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    run = runs[0]
    return run.info.run_id, run.data.metrics["f1_weighted"]


def promote(client: MlflowClient, model_name: str):
    # get the latest version of the model
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No registered versions found for model '{model_name}'")

    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, latest.version)
    print(f"Promoted {model_name} version {latest.version} to '{PRODUCTION_ALIAS}'")


def main():
    threshold = load_params()["f1_threshold"]
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
    model_name = os.getenv("MODEL_NAME", "finbert")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()

    run_id, f1 = get_latest_f1(client, experiment_name)
    print(f"Latest run {run_id}: f1_weighted={f1:.4f} (threshold={threshold})")

    if f1 >= threshold:
        promote(client, model_name)
    else:
        print(
            f"Model does not qualify for production: "
            f"f1_weighted {f1:.4f} < threshold {threshold}"
        )


if __name__ == "__main__":
    main()
