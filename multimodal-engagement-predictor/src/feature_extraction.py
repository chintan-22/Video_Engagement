"""Multimodal feature extraction with CLIP images and Gemma-style text scoring."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:
    from .config import CLIP_MODEL_NAME, GEMMA_MODEL_NAME, GEMMA_SCORE_KEYS
    from .utils import (
        caption_metadata_features,
        compact_lexical_vector,
        get_image_or_placeholder,
        sentiment_like_score,
        tokenize_caption,
    )
except ImportError:
    from config import CLIP_MODEL_NAME, GEMMA_MODEL_NAME, GEMMA_SCORE_KEYS
    from utils import (
        caption_metadata_features,
        compact_lexical_vector,
        get_image_or_placeholder,
        sentiment_like_score,
        tokenize_caption,
    )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _clamp_score(value: Any) -> int:
    try:
        return int(max(1, min(5, round(float(value)))))
    except (TypeError, ValueError):
        return 3


def handcrafted_image_embedding(image: Image.Image, dim: int = 512) -> np.ndarray:
    """Deterministic fallback image descriptor when CLIP is unavailable."""

    resized = image.resize((64, 64)).convert("RGB")
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)

    features: list[float] = []
    features.extend(arr.mean(axis=(0, 1)).tolist())
    features.extend(arr.std(axis=(0, 1)).tolist())
    features.extend(np.quantile(arr.reshape(-1, 3), [0.1, 0.25, 0.5, 0.75, 0.9], axis=0).ravel())

    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=32, range=(0.0, 1.0), density=True)
        features.extend(hist.astype(np.float32).tolist())

    gray_hist, _ = np.histogram(gray, bins=48, range=(0.0, 1.0), density=True)
    features.extend(gray_hist.astype(np.float32).tolist())

    horizontal_edges = np.abs(np.diff(gray, axis=1)).mean()
    vertical_edges = np.abs(np.diff(gray, axis=0)).mean()
    features.extend([float(horizontal_edges), float(vertical_edges)])

    base = np.array(features, dtype=np.float32)
    if base.size == 0:
        return np.zeros(dim, dtype=np.float32)

    tiled = np.resize(base, dim).astype(np.float32)
    norm = np.linalg.norm(tiled)
    if norm > 0:
        tiled /= norm
    return tiled


def compact_embedding(embedding: np.ndarray, dim: int = 16) -> np.ndarray:
    """Compress a large embedding into a fixed small vector by mean pooling chunks."""

    vector = np.asarray(embedding, dtype=np.float32).ravel()
    if vector.size == 0:
        return np.zeros(dim, dtype=np.float32)
    chunks = np.array_split(vector, dim)
    compact = np.array([chunk.mean() for chunk in chunks], dtype=np.float32)
    norm = np.linalg.norm(compact)
    if norm > 0:
        compact /= norm
    return compact


@dataclass
class ClipImageEncoder:
    """CLIP image encoder with a deterministic handcrafted fallback."""

    model_name: str = CLIP_MODEL_NAME
    mode: str = "auto"

    def __post_init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.source = "handcrafted"
        self.load_error: str | None = None

        if self.mode == "handcrafted":
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            local_only = _env_flag("CLIP_LOCAL_ONLY", False) or _env_flag("HF_LOCAL_FILES_ONLY", False)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )
            self.model.to(self.device)
            self.model.eval()
            self.source = "clip"
        except Exception as exc:
            self.load_error = str(exc)
            self.source = "handcrafted"
            if self.mode == "clip":
                print(f"Warning: CLIP could not be loaded, using handcrafted image features. {exc}")

    def encode_image(self, image: Image.Image) -> np.ndarray:
        if self.source != "clip":
            return handcrafted_image_embedding(image)

        import torch

        assert self.processor is not None
        assert self.model is not None

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy()[0].astype(np.float32)

    def encode_text_compact(self, caption: str, dim: int = 16) -> np.ndarray:
        """Use CLIP text features when available; otherwise use lexical hashing."""

        if self.source != "clip":
            return compact_lexical_vector(caption, dim=dim)

        import torch

        assert self.processor is not None
        assert self.model is not None

        inputs = self.processor(text=[caption], padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return compact_embedding(features.detach().cpu().numpy()[0], dim=dim)


class GemmaCaptionScorer:
    """Prompt-based Gemma scorer with a fast deterministic fallback.

    By default, the class tries a local-only Gemma load. Set ENABLE_GEMMA=1 to
    allow Hugging Face to download/load the configured Gemma model.
    """

    def __init__(self, model_name: str = GEMMA_MODEL_NAME, mode: str = "auto") -> None:
        self.model_name = model_name
        self.mode = mode
        self.source = "heuristic"
        self.load_error: str | None = None
        self.tokenizer = None
        self.model = None
        self.device = "cpu"

        if mode == "heuristic":
            return

        self._try_load_gemma()

    def _try_load_gemma(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            allow_download = _env_flag("ENABLE_GEMMA", False)
            local_only = (not allow_download) or _env_flag("HF_LOCAL_FILES_ONLY", False)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                local_files_only=local_only,
            )
            self.model.to(self.device)
            self.model.eval()
            self.source = "gemma"
        except Exception as exc:
            self.load_error = str(exc)
            self.source = "heuristic"

    def score_caption(self, caption: str) -> dict[str, int]:
        if self.source == "gemma":
            try:
                return self._score_with_gemma(caption)
            except Exception as exc:
                self.load_error = str(exc)
                self.source = "heuristic"
        return self._heuristic_scores(caption)

    def _score_with_gemma(self, caption: str) -> dict[str, int]:
        import torch

        assert self.tokenizer is not None
        assert self.model is not None

        prompt = f"""
You evaluate social-media caption engagement. Score each field from 1 (weak) to 5 (excellent).
Fields: hook_strength, emotional_appeal, clarity, curiosity_gap, cta_strength.
Caption: {caption!r}
Return only valid JSON with those five keys and integer values.
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return self._parse_scores(response)

    def _parse_scores(self, response: str) -> dict[str, int]:
        json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return {key: _clamp_score(parsed.get(key, 3)) for key in GEMMA_SCORE_KEYS}

        scores: dict[str, int] = {}
        for key in GEMMA_SCORE_KEYS:
            pattern = key.replace("_", r"[_\s-]*")
            match = re.search(pattern + r"[^0-9]{0,10}([1-5])", response, flags=re.IGNORECASE)
            scores[key] = _clamp_score(match.group(1)) if match else 3
        return scores

    def _heuristic_scores(self, caption: str) -> dict[str, int]:
        text = caption.strip()
        lower = text.lower()
        tokens = tokenize_caption(text)
        word_count = len(tokens)

        hook_terms = ("stop scrolling", "watch", "how to", "why", "secret", "before", "after")
        emotion_terms = ("love", "hate", "wow", "amazing", "stress", "win", "fail", "dream", "favorite")
        curiosity_terms = ("secret", "surprise", "guess", "what happened", "nobody", "hidden", "revealed")
        cta_terms = ("comment", "share", "save", "follow", "tag", "try", "click", "tell me", "join")

        hook = 2 + any(lower.startswith(term) or term in lower[:60] for term in hook_terms)
        hook += "?" in text or "!" in text

        emotional = 2 + sum(term in lower for term in emotion_terms)
        emotional += min(1, text.count("!") // 2)

        if 6 <= word_count <= 28:
            clarity = 5
        elif 3 <= word_count <= 40:
            clarity = 4
        elif word_count <= 60:
            clarity = 3
        else:
            clarity = 2
        if text.count("#") > 5:
            clarity -= 1

        curiosity = 2 + sum(term in lower for term in curiosity_terms)
        curiosity += "?" in text

        cta = 1 + sum(term in lower for term in cta_terms)
        cta += lower.endswith("?")

        return {
            "hook_strength": _clamp_score(hook),
            "emotional_appeal": _clamp_score(emotional),
            "clarity": _clamp_score(clarity),
            "curiosity_gap": _clamp_score(curiosity),
            "cta_strength": _clamp_score(cta),
        }


class MultimodalFeatureExtractor:
    """Create fused image + text feature vectors for training and inference."""

    def __init__(
        self,
        image_mode: str = "auto",
        text_mode: str = "auto",
        clip_model_name: str = CLIP_MODEL_NAME,
        gemma_model_name: str = GEMMA_MODEL_NAME,
    ) -> None:
        self.clip_encoder = ClipImageEncoder(model_name=clip_model_name, mode=image_mode)
        self.caption_scorer = GemmaCaptionScorer(model_name=gemma_model_name, mode=text_mode)
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> list[str]:
        return (
            [f"clip_image_{idx:03d}" for idx in range(512)]
            + [f"caption_score_{key}" for key in GEMMA_SCORE_KEYS]
            + [
                "caption_word_count_log",
                "caption_char_count_log",
                "caption_punctuation_rate",
                "caption_hashtag_count",
                "sentiment_like",
            ]
            + [f"compact_text_{idx:02d}" for idx in range(16)]
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "image_source": self.clip_encoder.source,
            "text_score_source": self.caption_scorer.source,
            "clip_model_name": self.clip_encoder.model_name,
            "gemma_model_name": self.caption_scorer.model_name,
            "clip_load_error": self.clip_encoder.load_error,
            "gemma_load_error": self.caption_scorer.load_error,
            "feature_count": len(self.feature_names),
        }

    def extract_one(
        self,
        image_path: str,
        caption: str,
        label: str | None = None,
        create_placeholder: bool = True,
    ) -> tuple[np.ndarray, dict[str, int]]:
        image = get_image_or_placeholder(
            image_path,
            caption=caption,
            label=label,
            create_if_missing=create_placeholder,
        )

        image_features = self.clip_encoder.encode_image(image)
        scores = self.caption_scorer.score_caption(caption)
        score_features = np.array([scores[key] / 5.0 for key in GEMMA_SCORE_KEYS], dtype=np.float32)
        metadata_features = caption_metadata_features(caption)
        sentiment_feature = np.array([sentiment_like_score(caption)], dtype=np.float32)
        compact_text = self.clip_encoder.encode_text_compact(caption, dim=16)

        fused = np.concatenate(
            [
                image_features,
                score_features,
                metadata_features,
                sentiment_feature,
                compact_text,
            ]
        ).astype(np.float32)

        return fused, scores

    def extract_dataframe_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, int]]]:
        features = []
        score_rows: list[dict[str, int]] = []

        for idx, row in df.reset_index(drop=True).iterrows():
            fused, scores = self.extract_one(
                image_path=row["image_path"],
                caption=row["caption"],
                label=row.get("engagement_label"),
                create_placeholder=True,
            )
            features.append(fused)
            score_rows.append(scores)
            if (idx + 1) % 10 == 0:
                print(f"Extracted features for {idx + 1}/{len(df)} samples")

        return np.vstack(features), score_rows
