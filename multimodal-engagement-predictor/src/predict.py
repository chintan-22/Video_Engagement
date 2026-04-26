"""Inference helpers for the Streamlit app and scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    from .config import GEMMA_SCORE_KEYS, MODEL_PATH
    from .feature_extraction import MultimodalFeatureExtractor
    from .utils import caption_metadata_features, sentiment_like_score
except ImportError:
    from config import GEMMA_SCORE_KEYS, MODEL_PATH
    from feature_extraction import MultimodalFeatureExtractor
    from utils import caption_metadata_features, sentiment_like_score


def load_model_bundle(model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run `python src/train.py` first."
        )
    return joblib.load(model_path)


def _build_extractor_from_bundle(bundle: dict[str, Any]) -> MultimodalFeatureExtractor:
    metadata = bundle.get("feature_metadata", {})
    image_mode = metadata.get("image_source", "auto")
    text_mode = metadata.get("text_score_source", "auto")

    if image_mode not in {"clip", "handcrafted"}:
        image_mode = "auto"
    if text_mode not in {"gemma", "heuristic"}:
        text_mode = "auto"

    return MultimodalFeatureExtractor(image_mode=image_mode, text_mode=text_mode)


def explain_prediction(
    label: str,
    caption: str,
    gemma_scores: dict[str, int],
    confidence: float,
) -> list[str]:
    """Create a compact, interview-friendly explanation of the prediction."""

    top_scores = sorted(gemma_scores.items(), key=lambda item: item[1], reverse=True)[:2]
    weak_scores = [name for name, value in gemma_scores.items() if value <= 2]
    metadata = caption_metadata_features(caption)
    sentiment = sentiment_like_score(caption)

    readable_top = ", ".join(name.replace("_", " ") for name, _ in top_scores)
    explanation = [
        f"The classifier predicted {label} with {confidence:.0%} confidence from fused image and caption features.",
        f"The strongest caption signals were {readable_top}.",
    ]

    if weak_scores:
        readable_weak = ", ".join(name.replace("_", " ") for name in weak_scores)
        explanation.append(f"Weaker caption signals included {readable_weak}.")

    if metadata[0] < 0.35:
        explanation.append("The caption is short, which can help clarity but may limit context.")
    elif metadata[0] > 0.75:
        explanation.append("The caption is relatively long, so clarity becomes more important.")

    if sentiment > 0.2:
        explanation.append("The wording has a positive sentiment-like signal.")
    elif sentiment < -0.2:
        explanation.append("The wording has a negative sentiment-like signal.")

    return explanation


def predict_engagement(image_path: str, caption: str) -> dict[str, Any]:
    """Predict High, Medium, or Low engagement for an image-caption pair."""

    bundle = load_model_bundle()
    classifier = bundle["model"]
    extractor = _build_extractor_from_bundle(bundle)

    features, gemma_scores = extractor.extract_one(
        image_path=image_path,
        caption=caption,
        label=None,
        create_placeholder=False,
    )

    X = features.reshape(1, -1)
    label = str(classifier.predict(X)[0])

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(X)[0]
        class_names = [str(class_name) for class_name in classifier.classes_]
        confidence = float(np.max(probabilities))
        probability_map = {
            class_name: float(probabilities[idx]) for idx, class_name in enumerate(class_names)
        }
    else:
        confidence = 1.0
        probability_map = {label: 1.0}

    return {
        "label": label,
        "confidence": confidence,
        "gemma_scores": {key: gemma_scores.get(key, 3) for key in GEMMA_SCORE_KEYS},
        "probabilities": probability_map,
        "feature_metadata": extractor.metadata,
        "explanation": explain_prediction(label, caption, gemma_scores, confidence),
    }
