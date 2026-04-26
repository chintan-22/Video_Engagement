"""Video retention-risk analysis for producer feedback.

This module is a lightweight prototype. It does not know true audience retention
without platform analytics; instead, it estimates likely drop-off points from
frame-level signals such as visual change, sharpness, brightness, contrast, and
opening-hook strength.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


@dataclass
class RetentionMoment:
    timestamp: str
    start_seconds: float
    end_seconds: float
    risk_score: float
    predicted_retention: float
    reasons: str
    recommendation: str


@dataclass
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""

    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _stable_name(url: str, suffix: str = ".mp4") -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"video_{digest}{suffix}"


def _is_probable_direct_video_url(url: str, content_type: str | None = None) -> bool:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return True
    if content_type and content_type.lower().startswith("video/"):
        return True
    return False


def _download_direct_video(
    url: str,
    output_base: Path,
    max_mb: int,
) -> Path:
    """Download a direct video file with browser-like headers."""

    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        suffix = ".mp4"

    output_path = output_base / _stable_name(url, suffix=suffix)
    headers = dict(DEFAULT_HTTP_HEADERS)
    headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"

    response = requests.get(url, stream=True, timeout=30, headers=headers)
    response.raise_for_status()

    max_bytes = max_mb * 1024 * 1024
    downloaded = 0
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > max_bytes:
                raise ValueError(f"Video is larger than the {max_mb} MB limit.")
            handle.write(chunk)
    return output_path


def _download_with_ytdlp(
    url: str,
    output_base: Path,
    cookies_browser: str | None = None,
) -> Path:
    """Download a URL with yt-dlp for video-page support."""

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "This URL looks like a video page rather than a direct video file. "
            "Install `yt-dlp` from requirements.txt, or use a direct .mp4/.webm/.mov URL."
        ) from exc

    output_template = str(output_base / "%(title).80s_%(id)s.%(ext)s")
    options: dict[str, Any] = {
        "format": "best[ext=mp4][vcodec!=none]/best[vcodec!=none]/best",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "geo_bypass": True,
        "http_headers": DEFAULT_HTTP_HEADERS,
        "extractor_args": {
            "youtube": {
                "player_client": ["ios", "android", "web"],
            }
        },
    }
    if cookies_browser:
        options["cookiesfrombrowser"] = (cookies_browser,)

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            downloaded_path = Path(downloader.prepare_filename(info))
            if not downloaded_path.exists():
                mp4_path = downloaded_path.with_suffix(".mp4")
                if mp4_path.exists():
                    downloaded_path = mp4_path
    except Exception as exc:
        message = str(exc)
        if "403" in message or "Forbidden" in message:
            raise RuntimeError(
                "The video host returned 403 Forbidden. That usually means the URL blocks automated downloads, "
                "requires login/cookies, is DRM-protected, or is an expiring signed link. Try uploading the video "
                "file directly, use a public direct .mp4/.webm/.mov URL, or enable browser cookies in the app."
            ) from exc
        raise RuntimeError(f"Could not download this video URL with yt-dlp: {message}") from exc

    if not downloaded_path.exists():
        raise FileNotFoundError("yt-dlp finished but the downloaded video file was not found.")
    return downloaded_path


def download_video_from_url(
    url: str,
    output_dir: str | Path | None = None,
    max_mb: int = 300,
    cookies_browser: str | None = None,
) -> Path:
    """Download a public video URL to a local temporary file.

    Direct video files are downloaded with requests. Page URLs such as YouTube
    can work when the optional `yt-dlp` dependency is installed.
    """

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Enter a public http or https video URL.")

    output_base = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="video_retention_"))
    output_base.mkdir(parents=True, exist_ok=True)

    head_type = None
    try:
        head = requests.head(url, allow_redirects=True, timeout=15, headers=DEFAULT_HTTP_HEADERS)
        head_type = head.headers.get("content-type")
    except requests.RequestException:
        head_type = None

    if _is_probable_direct_video_url(url, head_type):
        try:
            return _download_direct_video(url, output_base=output_base, max_mb=max_mb)
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 403:
                raise

    return _download_with_ytdlp(url, output_base=output_base, cookies_browser=cookies_browser)


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _get_ffmpeg_executable() -> str | None:
    """Find ffmpeg from the system or the bundled imageio-ffmpeg package."""

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _extract_audio_to_wav(video_path: Path, output_dir: Path) -> Path | None:
    """Extract mono 16 kHz WAV audio from a video if ffmpeg is available."""

    ffmpeg_path = _get_ffmpeg_executable()
    if not ffmpeg_path:
        return None

    wav_path = output_dir / f"{video_path.stem}_audio.wav"
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size == 0:
        return None
    return wav_path


def _read_wav_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file as float32 mono samples in [-1, 1]."""

    with wave.open(str(wav_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)

    if sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return samples.astype(np.float32), int(sample_rate)


def _audio_risk_and_reasons(
    audio_row: dict[str, float],
    timestamp: float,
    flat_audio_streak: int,
) -> tuple[float, list[str], list[str]]:
    risk = 12.0
    reasons: list[str] = []
    recommendations: list[str] = []

    if audio_row["silence_ratio"] > 0.72:
        risk += 28
        reasons.append("mostly silent audio")
        recommendations.append("Remove dead air or add voiceover, music, or sound design.")
    elif audio_row["audio_energy"] < 0.18:
        risk += 16
        reasons.append("low audio energy")
        recommendations.append("Lift dialogue/music levels or add a stronger audio cue.")

    if flat_audio_streak >= 3:
        risk += min(18, flat_audio_streak * 4)
        reasons.append("flat audio pattern")
        recommendations.append("Add vocal emphasis, music change, beat, or sound effect.")

    if audio_row["energy_drop"] > 0.45:
        risk += 13
        reasons.append("abrupt audio energy drop")
        recommendations.append("Smooth the audio transition or pair the drop with a clear visual payoff.")

    if audio_row["peak_level"] > 0.95:
        risk += 8
        reasons.append("possible audio clipping")
        recommendations.append("Reduce peaks or compress the mix.")

    if timestamp <= 8 and audio_row["audio_energy"] < 0.25:
        risk += 14
        reasons.append("weak audio hook")
        recommendations.append("Start with the strongest line, beat, or sound cue.")

    if audio_row["audio_energy"] > 0.55 and audio_row["energy_drop"] < 0.2:
        risk -= 5

    if not reasons:
        reasons.append("audio has usable energy")
        recommendations.append("Keep the audio momentum aligned with the visual beat.")

    return float(np.clip(risk, 0.0, 100.0)), reasons, recommendations


def _analyze_audio_timeline(
    video_path: Path,
    timestamps: np.ndarray,
    sample_every: float,
    temp_dir: Path,
) -> tuple[pd.DataFrame, str, Path | None]:
    """Extract and analyze audio energy, silence, and pacing over time."""

    wav_path = _extract_audio_to_wav(video_path, temp_dir)
    if wav_path is None:
        return (
            pd.DataFrame(),
            "Audio analysis skipped because ffmpeg was not available or no audio track was found.",
            None,
        )

    try:
        samples, sample_rate = _read_wav_mono(wav_path)
    except Exception as exc:
        return pd.DataFrame(), f"Audio analysis skipped because the extracted WAV could not be read: {exc}", wav_path

    if samples.size == 0:
        return pd.DataFrame(), "Audio analysis skipped because the audio track was empty.", wav_path

    raw_rows = []
    previous_rms: float | None = None
    for timestamp in timestamps:
        start = int(float(timestamp) * sample_rate)
        end = int(min(samples.size, (float(timestamp) + sample_every) * sample_rate))
        segment = samples[start:end]
        if segment.size == 0:
            continue

        rms = float(np.sqrt(np.mean(np.square(segment))))
        peak = float(np.max(np.abs(segment)))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(segment))).astype(np.float32))) if segment.size > 1 else 0.0
        raw_rows.append(
            {
                "timestamp_seconds": float(timestamp),
                "audio_rms": rms,
                "peak_level": peak,
                "zero_crossing_rate": zcr,
                "raw_energy_drop": max(0.0, (previous_rms or rms) - rms),
            }
        )
        previous_rms = rms

    if not raw_rows:
        return pd.DataFrame(), "Audio analysis skipped because no audio segments were readable.", wav_path

    audio_df = pd.DataFrame(raw_rows)
    low = float(audio_df["audio_rms"].quantile(0.1))
    high = float(audio_df["audio_rms"].quantile(0.9))
    silence_threshold = max(0.004, low * 0.75)

    rows = []
    flat_audio_streak = 0
    previous_energy = None

    for _, row in audio_df.iterrows():
        start = int(float(row["timestamp_seconds"]) * sample_rate)
        end = int(min(samples.size, (float(row["timestamp_seconds"]) + sample_every) * sample_rate))
        segment = samples[start:end]
        energy = _normalize(float(row["audio_rms"]), low, high if high > low else low + 0.01)
        silence_ratio = float(np.mean(np.abs(segment) < silence_threshold)) if segment.size else 1.0
        energy_drop = max(0.0, (previous_energy if previous_energy is not None else energy) - energy)
        flat_audio_streak = flat_audio_streak + 1 if abs(energy - (previous_energy or energy)) < 0.08 else 0

        audio_features = {
            "audio_energy": energy,
            "silence_ratio": silence_ratio,
            "energy_drop": energy_drop,
            "peak_level": float(row["peak_level"]),
            "zero_crossing_rate": float(row["zero_crossing_rate"]),
        }
        risk_score, reasons, recommendations = _audio_risk_and_reasons(
            audio_features,
            float(row["timestamp_seconds"]),
            flat_audio_streak,
        )
        rows.append(
            {
                "timestamp_seconds": float(row["timestamp_seconds"]),
                "audio_risk_score": risk_score,
                "audio_energy": energy,
                "silence_ratio": silence_ratio,
                "audio_energy_drop": energy_drop,
                "peak_level": float(row["peak_level"]),
                "zero_crossing_rate": float(row["zero_crossing_rate"]),
                "audio_reasons": ", ".join(reasons),
                "audio_recommendations": " | ".join(dict.fromkeys(recommendations)),
            }
        )
        previous_energy = energy

    return pd.DataFrame(rows), "Audio analysis used energy, silence, peaks, and audio-change signals.", wav_path


def _try_transcribe_audio(
    wav_path: Path,
    enabled: bool,
) -> tuple[list[TranscriptSegment], str]:
    """Optionally transcribe audio with a locally installed Whisper backend."""

    if not enabled:
        return [], "Transcript analysis disabled."

    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(wav_path), vad_filter=True)
        transcript = [
            TranscriptSegment(
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
                text=segment.text.strip(),
            )
            for segment in segments
            if segment.text.strip()
        ]
        return transcript, "Transcript analysis used faster-whisper base on CPU."
    except ImportError:
        pass
    except Exception as exc:
        return [], f"Transcript analysis failed with faster-whisper: {exc}"

    try:
        import whisper

        model = whisper.load_model("base")
        result = model.transcribe(str(wav_path), fp16=False)
        transcript = [
            TranscriptSegment(
                start_seconds=float(segment.get("start", 0.0)),
                end_seconds=float(segment.get("end", 0.0)),
                text=str(segment.get("text", "")).strip(),
            )
            for segment in result.get("segments", [])
            if str(segment.get("text", "")).strip()
        ]
        return transcript, "Transcript analysis used openai-whisper base on CPU."
    except ImportError:
        return [], "Transcript analysis skipped because faster-whisper/openai-whisper is not installed."
    except Exception as exc:
        return [], f"Transcript analysis failed with openai-whisper: {exc}"


def _transcript_text_for_time(segments: list[TranscriptSegment], timestamp: float, window: float) -> str:
    relevant = [
        segment.text
        for segment in segments
        if segment.end_seconds >= timestamp - window and segment.start_seconds <= timestamp + window
    ]
    return " ".join(relevant).strip()


def _analyze_transcript_timeline(
    wav_path: Path | None,
    timestamps: np.ndarray,
    sample_every: float,
    enabled: bool,
) -> tuple[pd.DataFrame, str, list[dict[str, Any]]]:
    if wav_path is None:
        return pd.DataFrame(), "Transcript analysis skipped because no audio WAV was available.", []

    segments, note = _try_transcribe_audio(wav_path, enabled=enabled)
    if not segments:
        return pd.DataFrame(), note, []

    try:
        from feature_extraction import GemmaCaptionScorer
    except ImportError:
        from .feature_extraction import GemmaCaptionScorer

    scorer = GemmaCaptionScorer(mode="auto")
    rows = []
    for timestamp in timestamps:
        text = _transcript_text_for_time(segments, float(timestamp), max(sample_every, 3.0))
        if not text:
            continue
        scores = scorer.score_caption(text)
        average_score = float(np.mean(list(scores.values())))
        transcript_risk = float(np.clip(100 - average_score * 20, 0, 100))
        if scores.get("clarity", 3) <= 2:
            transcript_risk += 10
        if scores.get("hook_strength", 3) <= 2 and timestamp <= 12:
            transcript_risk += 12
        rows.append(
            {
                "timestamp_seconds": float(timestamp),
                "transcript_risk_score": float(np.clip(transcript_risk, 0, 100)),
                "transcript_text": text[:240],
                "transcript_reasons": (
                    f"speech scored hook={scores.get('hook_strength', 3)}, "
                    f"emotion={scores.get('emotional_appeal', 3)}, "
                    f"clarity={scores.get('clarity', 3)}"
                ),
                "transcript_recommendations": "Tighten the spoken point, add a sharper payoff, or move the strongest line earlier.",
            }
        )

    return pd.DataFrame(rows), note, [asdict(segment) for segment in segments]


def _combine_modal_timelines(
    visual_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fuse visual, audio, and optional transcript risk into one timeline."""

    if visual_df.empty:
        return visual_df

    combined = visual_df.sort_values("timestamp_seconds").copy()
    combined["visual_risk_score"] = combined["risk_score"]

    if not audio_df.empty:
        combined = pd.merge_asof(
            combined,
            audio_df.sort_values("timestamp_seconds"),
            on="timestamp_seconds",
            direction="nearest",
            tolerance=max(0.01, float(np.diff(combined["timestamp_seconds"]).mean() if len(combined) > 1 else 1.0) / 2),
        )
    else:
        combined["audio_risk_score"] = np.nan

    if not transcript_df.empty:
        combined = pd.merge_asof(
            combined,
            transcript_df.sort_values("timestamp_seconds"),
            on="timestamp_seconds",
            direction="nearest",
            tolerance=max(3.0, float(np.diff(combined["timestamp_seconds"]).mean() if len(combined) > 1 else 1.0)),
        )
    else:
        combined["transcript_risk_score"] = np.nan

    has_audio = combined["audio_risk_score"].notna()
    has_transcript = combined["transcript_risk_score"].notna()

    combined["risk_score"] = combined["visual_risk_score"]
    combined.loc[has_audio, "risk_score"] = (
        combined.loc[has_audio, "visual_risk_score"] * 0.68
        + combined.loc[has_audio, "audio_risk_score"] * 0.32
    )
    combined.loc[has_audio & has_transcript, "risk_score"] = (
        combined.loc[has_audio & has_transcript, "visual_risk_score"] * 0.56
        + combined.loc[has_audio & has_transcript, "audio_risk_score"] * 0.26
        + combined.loc[has_audio & has_transcript, "transcript_risk_score"] * 0.18
    )
    combined.loc[~has_audio & has_transcript, "risk_score"] = (
        combined.loc[~has_audio & has_transcript, "visual_risk_score"] * 0.76
        + combined.loc[~has_audio & has_transcript, "transcript_risk_score"] * 0.24
    )
    combined["risk_score"] = combined["risk_score"].clip(0, 100)

    def join_texts(row: pd.Series, fields: list[str]) -> str:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, str) and value.strip() and value.strip().lower() != "nan":
                values.append(value.strip())
        return ", ".join(dict.fromkeys(values))

    def join_recommendations(row: pd.Series, fields: list[str]) -> str:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, str) and value.strip() and value.strip().lower() != "nan":
                values.extend(part.strip() for part in value.split(" | ") if part.strip())
        return " | ".join(dict.fromkeys(values))

    combined["reasons"] = combined.apply(
        lambda row: join_texts(row, ["visual_reasons", "audio_reasons", "transcript_reasons"]),
        axis=1,
    )
    combined["recommendations"] = combined.apply(
        lambda row: join_recommendations(
            row,
            ["visual_recommendations", "audio_recommendations", "transcript_recommendations"],
        ),
        axis=1,
    )
    return combined


def _entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=32, range=(0, 255), density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    entropy = -np.sum(hist * np.log2(hist))
    return float(np.clip(entropy / 5.0, 0.0, 1.0))


def _frame_metrics(frame: np.ndarray, previous_gray: np.ndarray | None) -> dict[str, float]:
    import cv2

    resized = cv2.resize(frame, (160, 90))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    brightness = float(gray.mean() / 255.0)
    contrast = _normalize(float(gray.std()), 12.0, 75.0)
    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = _normalize(math.log1p(sharpness_raw), 2.8, 6.4)
    saturation = float(hsv[:, :, 1].mean() / 255.0)
    entropy = _entropy(gray)

    if previous_gray is None:
        visual_change = 0.08
    else:
        visual_change = float(np.mean(np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32))) / 255.0)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "saturation": saturation,
        "entropy": entropy,
        "visual_change": visual_change,
        "_gray": gray,
    }


def _risk_and_reasons(row: dict[str, float], timestamp: float, static_streak: int) -> tuple[float, list[str], list[str]]:
    risk = 18.0
    reasons: list[str] = []
    recommendations: list[str] = []

    if row["visual_change"] < 0.015:
        risk += 24
        reasons.append("very little scene or motion change")
        recommendations.append("Add a cutaway, zoom, overlay, or pattern interrupt.")
    elif row["visual_change"] < 0.035:
        risk += 13
        reasons.append("low visual change")
        recommendations.append("Shorten the static section or add B-roll.")
    elif row["visual_change"] > 0.12:
        risk -= 7

    if static_streak >= 3:
        risk += min(18, static_streak * 4)
        reasons.append("sustained static stretch")
        recommendations.append("Break the stretch into tighter beats.")

    if row["sharpness"] < 0.25:
        risk += 17
        reasons.append("soft or blurry frame")
        recommendations.append("Replace blurry shots or add sharper supporting visuals.")
    elif row["sharpness"] < 0.42:
        risk += 8
        reasons.append("moderate sharpness")

    if row["brightness"] < 0.16:
        risk += 13
        reasons.append("dark frame")
        recommendations.append("Lift exposure or use a brighter close-up.")
    elif row["brightness"] > 0.88:
        risk += 10
        reasons.append("over-bright frame")
        recommendations.append("Reduce highlights so detail is easier to read.")

    if row["contrast"] < 0.2:
        risk += 11
        reasons.append("low contrast")
        recommendations.append("Increase contrast or separate subject from background.")

    if row["saturation"] < 0.12:
        risk += 7
        reasons.append("muted color")

    if row["entropy"] < 0.35:
        risk += 9
        reasons.append("low visual detail")
        recommendations.append("Show a more information-rich shot or visual proof.")

    if timestamp <= 8 and row["visual_change"] < 0.045:
        risk += 18
        reasons.append("weak opening hook")
        recommendations.append("Move the outcome, conflict, or strongest visual into the first 3 seconds.")

    if timestamp <= 15 and row["brightness"] < 0.2:
        risk += 8
        reasons.append("opening is visually hard to read")

    if not reasons:
        reasons.append("stable segment with no major quality issue")
        recommendations.append("Keep this pacing; use it as a baseline for weaker sections.")

    return float(np.clip(risk, 0.0, 100.0)), reasons, recommendations


def _add_retention_curve(frame_df: pd.DataFrame, duration: float) -> pd.DataFrame:
    if frame_df.empty:
        frame_df["predicted_retention"] = []
        return frame_df

    working = frame_df.copy()
    early_weight = np.where(working["timestamp_seconds"] <= 12, 1.55, 1.0)
    risk_weight = np.maximum(working["risk_score"].to_numpy(), 1.0) * early_weight
    cumulative = np.cumsum(risk_weight)
    total_weight = max(float(cumulative[-1]), 1.0)
    average_risk = float(working["risk_score"].mean())
    expected_total_drop = float(np.clip(28 + average_risk * 0.55 + min(duration / 60.0, 4) * 3, 35, 82))
    working["predicted_retention"] = 100.0 - (cumulative / total_weight) * expected_total_drop
    working["predicted_retention"] = working["predicted_retention"].clip(lower=8.0, upper=100.0)
    return working


def _opening_diagnostics(frame_df: pd.DataFrame, opening_seconds: float) -> dict[str, Any] | None:
    """Summarize first-impression issues separately from later drop-off moments."""

    if frame_df.empty:
        return None

    opening_df = frame_df[frame_df["timestamp_seconds"] <= opening_seconds].copy()
    if opening_df.empty:
        return None

    peak = opening_df.sort_values("risk_score", ascending=False).iloc[0]
    average_risk = float(opening_df["risk_score"].mean())
    if average_risk < 45 and float(peak["risk_score"]) < 55:
        return None

    recommendations = []
    for value in opening_df["recommendations"].tolist():
        recommendations.extend(str(value).split(" | "))

    unique_recommendations = []
    for item in recommendations:
        if item and item not in unique_recommendations:
            unique_recommendations.append(item)

    return {
        "timestamp": f"00:00-{format_timestamp(opening_seconds)}",
        "risk_score": average_risk,
        "peak_risk_score": float(peak["risk_score"]),
        "reasons": str(peak["reasons"]),
        "recommendation": " ".join(unique_recommendations[:3]),
        "note": "Opening diagnostics are first-impression issues, not a viewer drop-off timestamp.",
    }


def _group_loss_moments(
    frame_df: pd.DataFrame,
    sample_every: float,
    min_loss_seconds: float,
) -> list[RetentionMoment]:
    if frame_df.empty:
        return []

    analysis_df = frame_df[frame_df["timestamp_seconds"] >= min_loss_seconds].copy()
    if analysis_df.empty:
        return []

    threshold = max(58.0, float(frame_df["risk_score"].quantile(0.75)))
    candidate_df = analysis_df[analysis_df["risk_score"] >= threshold].copy()
    if candidate_df.empty:
        fallback_threshold = max(52.0, float(analysis_df["risk_score"].quantile(0.7)))
        candidate_df = analysis_df[analysis_df["risk_score"] >= fallback_threshold].copy()

    if candidate_df.empty:
        return []

    groups: list[pd.DataFrame] = []
    current_rows: list[pd.Series] = []
    previous_time: float | None = None

    for _, row in candidate_df.sort_values("timestamp_seconds").iterrows():
        current_time = float(row["timestamp_seconds"])
        if previous_time is None or current_time - previous_time <= sample_every * 1.5:
            current_rows.append(row)
        else:
            groups.append(pd.DataFrame(current_rows))
            current_rows = [row]
        previous_time = current_time

    if current_rows:
        groups.append(pd.DataFrame(current_rows))

    moments: list[RetentionMoment] = []
    for group in groups:
        peak = group.sort_values("risk_score", ascending=False).iloc[0]
        start = float(group["timestamp_seconds"].min())
        end = float(group["timestamp_seconds"].max() + sample_every)
        recommendations = []
        for value in group["recommendations"].tolist():
            recommendations.extend(str(value).split(" | "))
        unique_recommendations = []
        for item in recommendations:
            if item and item not in unique_recommendations:
                unique_recommendations.append(item)

        moments.append(
            RetentionMoment(
                timestamp=format_timestamp(float(peak["timestamp_seconds"])),
                start_seconds=start,
                end_seconds=end,
                risk_score=float(peak["risk_score"]),
                predicted_retention=float(peak["predicted_retention"]),
                reasons=str(peak["reasons"]),
                recommendation=" ".join(unique_recommendations[:3]),
            )
        )

    moments = sorted(moments, key=lambda item: item.risk_score, reverse=True)[:6]
    return sorted(moments, key=lambda item: item.start_seconds)


def analyze_video_file(
    video_path: str | Path,
    sample_every: float = 2.0,
    max_samples: int = 180,
    include_audio: bool = True,
    include_transcript: bool = False,
) -> dict[str, Any]:
    """Analyze a video file and return timeline analytics."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video analysis. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Could not open the video. Try a different URL or file format.")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    if duration <= 0:
        capture.release()
        raise ValueError("Could not determine the video duration.")

    adjusted_interval = max(float(sample_every), duration / max(max_samples, 1))
    timestamps = np.arange(0, duration, adjusted_interval)

    rows = []
    previous_gray = None
    static_streak = 0

    for timestamp in timestamps:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        metrics = _frame_metrics(frame, previous_gray)
        previous_gray = metrics.pop("_gray")
        static_streak = static_streak + 1 if metrics["visual_change"] < 0.035 else 0
        risk_score, reasons, recommendations = _risk_and_reasons(metrics, float(timestamp), static_streak)

        rows.append(
            {
                "timestamp": format_timestamp(float(timestamp)),
                "timestamp_seconds": float(timestamp),
                "risk_score": risk_score,
                "visual_risk_score": risk_score,
                "brightness": metrics["brightness"],
                "contrast": metrics["contrast"],
                "sharpness": metrics["sharpness"],
                "saturation": metrics["saturation"],
                "visual_change": metrics["visual_change"],
                "visual_detail": metrics["entropy"],
                "visual_reasons": ", ".join(reasons),
                "visual_recommendations": " | ".join(dict.fromkeys(recommendations)),
                "reasons": ", ".join(reasons),
                "recommendations": " | ".join(dict.fromkeys(recommendations)),
            }
        )

    capture.release()

    frame_df = pd.DataFrame(rows)
    audio_df = pd.DataFrame()
    transcript_df = pd.DataFrame()
    transcript_segments: list[dict[str, Any]] = []
    modality_notes = []

    with tempfile.TemporaryDirectory(prefix="video_modal_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        wav_path = None
        if include_audio or include_transcript:
            audio_df, audio_note, wav_path = _analyze_audio_timeline(
                video_path,
                timestamps,
                adjusted_interval,
                tmp_dir_path,
            )
            modality_notes.append(audio_note)
        else:
            modality_notes.append("Audio analysis disabled.")

        if include_transcript:
            transcript_df, transcript_note, transcript_segments = _analyze_transcript_timeline(
                wav_path,
                timestamps,
                adjusted_interval,
                enabled=True,
            )
            modality_notes.append(transcript_note)
        else:
            modality_notes.append("Transcript analysis disabled.")

        frame_df = _combine_modal_timelines(frame_df, audio_df, transcript_df)

    frame_df = _add_retention_curve(frame_df, duration)
    opening_window_seconds = max(3.0, adjusted_interval)
    opening_summary = _opening_diagnostics(frame_df, opening_window_seconds)
    moments = _group_loss_moments(
        frame_df,
        adjusted_interval,
        min_loss_seconds=opening_window_seconds,
    )

    average_risk = float(frame_df["risk_score"].mean()) if not frame_df.empty else 0.0
    audio_available = "audio_risk_score" in frame_df.columns and frame_df["audio_risk_score"].notna().any()
    transcript_available = (
        "transcript_risk_score" in frame_df.columns and frame_df["transcript_risk_score"].notna().any()
    )
    audio_average_risk = (
        float(frame_df["audio_risk_score"].dropna().mean())
        if audio_available
        else None
    )
    transcript_average_risk = (
        float(frame_df["transcript_risk_score"].dropna().mean())
        if transcript_available
        else None
    )
    completion = float(frame_df["predicted_retention"].iloc[-1]) if not frame_df.empty else 0.0
    post_opening_df = frame_df[frame_df["timestamp_seconds"] >= opening_window_seconds]
    riskiest = (
        post_opening_df.sort_values("risk_score", ascending=False).iloc[0].to_dict()
        if not post_opening_df.empty
        else {}
    )
    early_df = frame_df[frame_df["timestamp_seconds"] <= 8]
    hook_score = float(np.clip(100 - early_df["risk_score"].mean(), 0, 100)) if not early_df.empty else 0.0

    summary = {
        "duration_seconds": float(duration),
        "duration": format_timestamp(duration),
        "fps": fps,
        "sample_every_seconds": float(adjusted_interval),
        "sample_count": int(len(frame_df)),
        "average_risk": average_risk,
        "audio_available": bool(audio_available),
        "audio_average_risk": audio_average_risk,
        "transcript_available": bool(transcript_available),
        "transcript_average_risk": transcript_average_risk,
        "predicted_completion_rate": completion,
        "hook_score": hook_score,
        "opening_window_seconds": opening_window_seconds,
        "riskiest_timestamp": str(riskiest.get("timestamp", "N/A")),
        "riskiest_risk_score": float(riskiest.get("risk_score", 0.0)),
    }

    return {
        "summary": summary,
        "timeline": frame_df,
        "opening_diagnostics": opening_summary,
        "loss_moments": [asdict(moment) for moment in moments],
        "transcript_segments": transcript_segments,
        "modality_notes": modality_notes,
        "method_note": (
            "This is a prototype retention-risk estimate based on visual pacing, audio energy, "
            "and optional transcript signals. "
            "True retention requires platform watch-time data."
        ),
    }


def analyze_video_url(
    url: str,
    sample_every: float = 2.0,
    max_samples: int = 180,
    max_mb: int = 300,
    cookies_browser: str | None = None,
    include_audio: bool = True,
    include_transcript: bool = False,
) -> dict[str, Any]:
    """Download a video URL and analyze its likely retention-risk moments."""

    with tempfile.TemporaryDirectory(prefix="video_retention_") as tmp_dir:
        video_path = download_video_from_url(
            url,
            output_dir=tmp_dir,
            max_mb=max_mb,
            cookies_browser=cookies_browser,
        )
        result = analyze_video_file(
            video_path,
            sample_every=sample_every,
            max_samples=max_samples,
            include_audio=include_audio,
            include_transcript=include_transcript,
        )
        result["source"] = {"url": url, "downloaded_path": str(video_path)}
        return result
