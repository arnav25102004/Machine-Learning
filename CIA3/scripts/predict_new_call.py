"""
Load the saved triage pipeline and predict the urgency class for a new synthetic call record.

Usage:
    python scripts/predict_new_call.py
"""
import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "triage_pipeline.joblib")


def load_pipeline():
    return joblib.load(MODEL_PATH)


def predict_call(record: dict, artifact=None):
    """record must contain exactly the caller-obtainable fields the pipeline expects."""
    if artifact is None:
        artifact = load_pipeline()
    pipe = artifact["pipeline"]
    cols = artifact["feature_columns"]

    row = pd.DataFrame([record])[cols]
    pred = pipe.predict(row)[0]
    proba = dict(zip(pipe.classes_, pipe.predict_proba(row)[0]))
    return pred, proba


if __name__ == "__main__":
    example_call = {
        "Age": 72,
        "NRS_pain": 8,
        "Sex": "Female",
        "Arrival mode": "Private Ambulance",
        "Injury": "No",
        "Mental": "Verbal Response",
        "Chief_complain_group": "Respiratory",
        "Caller_relationship": "Family member",
    }

    artifact = load_pipeline()
    print("Loaded model:", artifact["model_name"])
    print("Expected feature columns:", artifact["feature_columns"])
    print()

    pred, proba = predict_call(example_call, artifact)
    print("Input call record:", example_call)
    print()
    print("Predicted triage class:", pred)
    print("Class probabilities:")
    for cls, p in sorted(proba.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:12s}: {p:.3f}")
