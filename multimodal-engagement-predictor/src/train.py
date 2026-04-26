"""Train a lightweight classifier on fused multimodal features."""

from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

try:
    from .config import LABEL_ORDER, METRICS_PATH, MODEL_PATH, RANDOM_SEED, SAMPLE_DATA_PATH, TEST_SIZE
    from .feature_extraction import MultimodalFeatureExtractor
    from .utils import ensure_directories, load_sample_data, save_json
except ImportError:
    from config import LABEL_ORDER, METRICS_PATH, MODEL_PATH, RANDOM_SEED, SAMPLE_DATA_PATH, TEST_SIZE
    from feature_extraction import MultimodalFeatureExtractor
    from utils import ensure_directories, load_sample_data, save_json


def train_model() -> dict:
    ensure_directories()

    df = load_sample_data(SAMPLE_DATA_PATH)
    print(f"Loaded {len(df)} samples from {SAMPLE_DATA_PATH}")

    extractor = MultimodalFeatureExtractor()
    X, text_scores = extractor.extract_dataframe_features(df)
    y = df["engagement_label"].values

    np.save(MODEL_PATH.parent / "fused_features.npy", X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        labels=LABEL_ORDER,
        zero_division=0,
        output_dict=True,
    )

    print("\nFeature extraction metadata:")
    print(json.dumps(extractor.metadata, indent=2))
    print(f"\nAccuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=LABEL_ORDER,
            zero_division=0,
        )
    )

    bundle = {
        "model": classifier,
        "feature_metadata": extractor.metadata,
        "feature_names": extractor.feature_names,
        "label_order": LABEL_ORDER,
        "text_scores_sample": text_scores[:5],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "feature_metadata": extractor.metadata,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
    }
    save_json(METRICS_PATH, metrics)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    train_model()
