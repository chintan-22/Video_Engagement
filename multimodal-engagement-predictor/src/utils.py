"""Utility helpers for data loading, placeholder images, and simple text signals."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    from .config import ARTIFACTS_DIR, DATA_DIR, IMAGE_DIR, ROOT_DIR
except ImportError:
    from config import ARTIFACTS_DIR, DATA_DIR, IMAGE_DIR, ROOT_DIR


POSITIVE_WORDS = {
    "amazing",
    "best",
    "boost",
    "bright",
    "calm",
    "easy",
    "favorite",
    "free",
    "fresh",
    "glow",
    "growth",
    "happy",
    "love",
    "new",
    "powerful",
    "simple",
    "smart",
    "win",
}

NEGATIVE_WORDS = {
    "bad",
    "boring",
    "confusing",
    "fail",
    "hard",
    "miss",
    "problem",
    "slow",
    "stress",
    "struggle",
    "tired",
    "waste",
    "worst",
}


def ensure_directories() -> None:
    """Create project runtime directories if they do not exist."""

    for directory in (DATA_DIR, IMAGE_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def stable_int(value: str, modulo: int | None = None) -> int:
    """Return a deterministic integer hash for a string."""

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    number = int(digest[:16], 16)
    return number % modulo if modulo else number


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path that may be absolute or relative to the project root."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_sample_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate the sample engagement dataset."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"image_path", "caption", "likes", "engagement_label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["caption"] = df["caption"].fillna("").astype(str)
    df["engagement_label"] = df["engagement_label"].astype(str)
    return df


def create_placeholder_image(
    image_path: Path,
    caption: str = "",
    label: str | None = None,
    size: tuple[int, int] = (384, 384),
) -> Path:
    """Create a deterministic synthetic image for a missing sample.

    The real project path expects social images under data/images. To keep this
    prototype small, training can generate simple placeholders that still let the
    image feature pipeline run end to end.
    """

    image_path.parent.mkdir(parents=True, exist_ok=True)

    seed_text = f"{image_path.name}|{caption}|{label or ''}"
    rng = np.random.default_rng(stable_int(seed_text, 2**32))

    palettes = {
        "High": ((238, 84, 64), (255, 208, 98), (34, 151, 128)),
        "Medium": ((64, 121, 191), (246, 191, 96), (236, 240, 241)),
        "Low": ((84, 91, 110), (166, 169, 179), (224, 224, 224)),
    }
    colors = palettes.get(label or "", ((90, 130, 180), (230, 230, 230), (40, 60, 80)))

    image = Image.new("RGB", size, colors[0])
    draw = ImageDraw.Draw(image)

    for _ in range(9):
        x0 = int(rng.integers(0, size[0] - 60))
        y0 = int(rng.integers(0, size[1] - 60))
        w = int(rng.integers(40, 160))
        h = int(rng.integers(40, 160))
        color = colors[int(rng.integers(0, len(colors)))]
        if rng.random() > 0.45:
            draw.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=18, fill=color)
        else:
            draw.ellipse((x0, y0, x0 + w, y0 + h), fill=color)

    title = (label or "Sample").upper()
    words = re.findall(r"[A-Za-z0-9#']+", caption)[:5]
    subtitle = " ".join(words) if words else "placeholder image"

    try:
        font_large = ImageFont.truetype("Arial.ttf", 34)
        font_small = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle((0, size[1] - 105, size[0], size[1]), fill=(0, 0, 0, 145))
    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(image)
    draw.text((24, size[1] - 88), title, fill=(255, 255, 255), font=font_large)
    draw.text((24, size[1] - 42), subtitle[:42], fill=(235, 235, 235), font=font_small)

    image.convert("RGB").save(image_path, quality=92)
    return image_path


def get_image_or_placeholder(
    image_path_value: str | Path,
    caption: str = "",
    label: str | None = None,
    create_if_missing: bool = True,
) -> Image.Image:
    """Load an image as RGB, optionally generating a deterministic placeholder."""

    image_path = resolve_project_path(image_path_value)
    if not image_path.exists() and create_if_missing:
        create_placeholder_image(image_path, caption=caption, label=label)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}. Add the image or enable placeholder generation."
        )

    try:
        return Image.open(image_path).convert("RGB")
    except Exception as exc:
        if not create_if_missing:
            raise
        fallback_path = IMAGE_DIR / f"fallback_{stable_int(str(image_path), 10_000)}.jpg"
        create_placeholder_image(fallback_path, caption=caption, label=label)
        print(f"Warning: could not read {image_path}: {exc}. Using {fallback_path}.")
        return Image.open(fallback_path).convert("RGB")


def tokenize_caption(caption: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z']+", caption.lower())


def sentiment_like_score(caption: str) -> float:
    """A tiny lexicon score in [-1, 1] for fast, interpretable signal."""

    tokens = tokenize_caption(caption)
    if not tokens:
        return 0.0
    positive = sum(token in POSITIVE_WORDS for token in tokens)
    negative = sum(token in NEGATIVE_WORDS for token in tokens)
    return float((positive - negative) / math.sqrt(len(tokens)))


def caption_metadata_features(caption: str) -> np.ndarray:
    tokens = tokenize_caption(caption)
    word_count = len(tokens)
    char_count = len(caption)
    punctuation_rate = (caption.count("!") + caption.count("?")) / max(word_count, 1)
    hashtag_count = caption.count("#")

    return np.array(
        [
            math.log1p(word_count) / 5.0,
            math.log1p(char_count) / 7.0,
            min(punctuation_rate, 1.0),
            min(hashtag_count / 5.0, 1.0),
        ],
        dtype=np.float32,
    )


def compact_lexical_vector(caption: str, dim: int = 16) -> np.ndarray:
    """Hash caption tokens into a compact deterministic vector."""

    vector = np.zeros(dim, dtype=np.float32)
    for token in tokenize_caption(caption):
        bucket = stable_int(token, dim)
        sign = 1.0 if stable_int(f"{token}:sign", 2) == 0 else -1.0
        vector[bucket] += sign

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector
