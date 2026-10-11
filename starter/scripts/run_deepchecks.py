"""
Model quality check (drift) using deepchecks

Run:
    python scripts/run_deepchecks.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yaml
from deepchecks.nlp import TextData
from deepchecks.nlp.checks import PredictionDrift, PropertyDrift
from dotenv import load_dotenv

from app.utils import load_classifier

load_dotenv()


MODEL_CLASSES = ["negative", "neutral", "positive"]


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["deepchecks"]


def run_predictions(classifier, texts: list[str]) -> list[str]:
    results = classifier(texts, batch_size=32)
    return [r["label"] for r in results]


def main():
    params = load_params()
    property_drift_threshold = params["property_drift_threshold"]
    prediction_drift_threshold = params["prediction_drift_threshold"]

    print("Loading production model...")

    classifier = load_classifier()

    stream_df = pd.read_csv("data/stream.csv")
    test_df = pd.read_csv("data/test.csv")

    stream_texts = stream_df["text"].tolist()
    test_texts = test_df["text"].tolist()

    test_dataset = TextData(
        raw_text=test_texts, task_type="text_classification", name="test"
    )
    stream_dataset = TextData(
        raw_text=stream_texts, task_type="text_classification", name="stream"
    )
    test_dataset.calculate_builtin_properties(include_properties=["Sentiment"])
    stream_dataset.calculate_builtin_properties(include_properties=["Sentiment"])

    # Property drift: test set is the reference, stream is the current data
    property_result = PropertyDrift(properties=["Sentiment"]).run(
        train_dataset=test_dataset, test_dataset=stream_dataset, with_display=False
    )
    property_drift = property_result.value["Sentiment"]["Drift score"]
    print(
        f"Sentiment property drift: {property_drift:.4f} "
        f"(threshold={property_drift_threshold})"
    )
    if property_drift > property_drift_threshold:
        print("ERROR: Sentiment property drift exceeds threshold")
        sys.exit(1)

    # Prediction drift between model outputs on both datasets
    test_predictions = run_predictions(classifier, test_texts)
    stream_predictions = run_predictions(classifier, stream_texts)

    prediction_check = PredictionDrift().add_condition_drift_score_less_than(
        max_allowed_drift_score=prediction_drift_threshold
    )
    prediction_result = prediction_check.run(
        train_dataset=test_dataset,
        test_dataset=stream_dataset,
        train_predictions=test_predictions,
        test_predictions=stream_predictions,
        model_classes=MODEL_CLASSES,
        with_display=False,
    )
    prediction_drift = prediction_result.value["Drift score"]
    print(
        f"Prediction drift: {prediction_drift:.4f} "
        f"(threshold={prediction_drift_threshold})"
    )
    if not prediction_result.passed_conditions():
        print("ERROR: Prediction drift exceeds threshold")
        sys.exit(1)

    print("All drift checks passed.")


if __name__ == "__main__":
    main()
