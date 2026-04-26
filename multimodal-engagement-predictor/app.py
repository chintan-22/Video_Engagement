"""Streamlit demos for multimodal engagement and video retention analysis."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import predict_engagement  # noqa: E402
from video_analysis import analyze_video_file, analyze_video_url  # noqa: E402


st.set_page_config(page_title="Multimodal Engagement Studio", layout="wide")


def render_image_caption_tab() -> None:
    st.header("Image + Caption Engagement Predictor")
    st.caption("Interview prototype: CLIP image embeddings plus Gemma-style caption scoring.")

    uploaded_file = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])
    caption = st.text_area(
        "Caption",
        placeholder="Stop scrolling: this 5-minute workflow changed my whole afternoon.",
        height=120,
    )

    if uploaded_file is not None:
        preview = Image.open(uploaded_file).convert("RGB")
        st.image(preview, use_container_width=True)

    predict_button = st.button("Predict Engagement", type="primary")

    if predict_button:
        if uploaded_file is None:
            st.warning("Upload an image first.")
        elif not caption.strip():
            st.warning("Enter a caption first.")
        else:
            suffix = Path(uploaded_file.name).suffix or ".jpg"
            tmp_path = None

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                with st.spinner("Extracting fused features"):
                    result = predict_engagement(tmp_path, caption)

                col_a, col_b = st.columns(2)
                col_a.metric("Prediction", result["label"])
                col_b.metric("Confidence", f"{result['confidence']:.0%}")

                st.subheader("Caption Scores")
                score_frame = pd.DataFrame(
                    [
                        {
                            "signal": key.replace("_", " ").title(),
                            "score": value,
                        }
                        for key, value in result["gemma_scores"].items()
                    ]
                )
                st.dataframe(score_frame, hide_index=True, use_container_width=True)

                st.subheader("Class Probabilities")
                probability_frame = pd.DataFrame(
                    [
                        {"label": label, "probability": probability}
                        for label, probability in result["probabilities"].items()
                    ]
                )
                st.bar_chart(probability_frame.set_index("label"))

                st.subheader("Signals")
                for item in result["explanation"]:
                    st.write(item)

                metadata = result.get("feature_metadata", {})
                st.caption(
                    f"Image encoder: {metadata.get('image_source', 'unknown')} | "
                    f"Text scorer: {metadata.get('text_score_source', 'unknown')}"
                )
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.info("Run `python src/train.py` from the project folder, then reload the app.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)


def render_video_results(result: dict) -> None:
    summary = result["summary"]
    timeline = result["timeline"].copy()
    loss_moments = pd.DataFrame(result["loss_moments"])

    st.info(result["method_note"])
    for note in result.get("modality_notes", []):
        st.caption(note)

    col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
    col_a.metric("Duration", summary["duration"])
    col_b.metric("Avg Risk", f"{summary['average_risk']:.0f}/100")
    col_c.metric("Hook Score", f"{summary['hook_score']:.0f}/100")
    if summary.get("audio_available"):
        col_d.metric("Audio Risk", f"{summary['audio_average_risk']:.0f}/100")
    else:
        col_d.metric("Audio Risk", "N/A")
    col_e.metric("Predicted Completion", f"{summary['predicted_completion_rate']:.0f}%")
    col_f.metric("Riskiest Moment", summary["riskiest_timestamp"])

    st.subheader("Retention Risk Timeline")
    chart_df = timeline[
        ["timestamp_seconds", "risk_score", "predicted_retention"]
    ].rename(
        columns={
            "timestamp_seconds": "seconds",
            "risk_score": "Drop-off risk",
            "predicted_retention": "Predicted retention",
        }
    )
    st.line_chart(chart_df, x="seconds", y=["Drop-off risk", "Predicted retention"], height=340)

    modal_columns = ["visual_risk_score", "audio_risk_score", "transcript_risk_score"]
    available_modal_columns = [column for column in modal_columns if column in timeline.columns and timeline[column].notna().any()]
    if available_modal_columns:
        st.subheader("Modal Risk Breakdown")
        modal_chart = timeline[["timestamp_seconds"] + available_modal_columns].rename(
            columns={
                "timestamp_seconds": "seconds",
                "visual_risk_score": "Visual risk",
                "audio_risk_score": "Audio risk",
                "transcript_risk_score": "Transcript risk",
            }
        )
        st.line_chart(modal_chart, x="seconds", y=list(modal_chart.columns[1:]), height=260)

    st.subheader("Where Viewers May Lose Interest")
    if loss_moments.empty:
        st.success("No major risk spikes were detected.")
    else:
        display_moments = loss_moments[
            [
                "timestamp",
                "risk_score",
                "predicted_retention",
                "reasons",
                "recommendation",
            ]
        ].rename(
            columns={
                "timestamp": "timestamp",
                "risk_score": "risk score",
                "predicted_retention": "predicted retention",
                "reasons": "why it may drop",
                "recommendation": "producer fix",
            }
        )
        display_moments["risk score"] = display_moments["risk score"].round(1)
        display_moments["predicted retention"] = display_moments["predicted retention"].round(1)
        st.dataframe(display_moments, hide_index=True, use_container_width=True)

    st.subheader("Producer Action Plan")
    if not loss_moments.empty:
        for _, row in loss_moments.iterrows():
            st.write(
                f"At **{row['timestamp']}**, risk rises because of {row['reasons']}. "
                f"{row['recommendation']}"
            )
    else:
        st.write("The video has a steady visual rhythm. Review the detailed timeline for smaller refinements.")

    with st.expander("Detailed Frame-Level Analytics"):
        requested_columns = [
            "timestamp",
            "risk_score",
            "predicted_retention",
            "visual_risk_score",
            "audio_risk_score",
            "transcript_risk_score",
            "visual_change",
            "audio_energy",
            "silence_ratio",
            "audio_energy_drop",
            "brightness",
            "contrast",
            "sharpness",
            "saturation",
            "visual_detail",
            "transcript_text",
            "reasons",
        ]
        detailed_columns = [column for column in requested_columns if column in timeline.columns]
        detailed = timeline[detailed_columns].copy()
        numeric_cols = detailed.select_dtypes(include="number").columns
        detailed[numeric_cols] = detailed[numeric_cols].round(3)
        st.dataframe(detailed, hide_index=True, use_container_width=True)

    transcript_segments = pd.DataFrame(result.get("transcript_segments", []))
    if not transcript_segments.empty:
        with st.expander("Transcript Segments"):
            transcript_segments["start"] = transcript_segments["start_seconds"].map(lambda value: f"{value:.1f}s")
            transcript_segments["end"] = transcript_segments["end_seconds"].map(lambda value: f"{value:.1f}s")
            st.dataframe(transcript_segments[["start", "end", "text"]], hide_index=True, use_container_width=True)


def render_video_retention_tab() -> None:
    st.header("Video Retention Analyzer")
    st.caption(
        "Paste a public video URL or upload a video. The app estimates likely viewer drop-off moments "
        "from visual pacing and frame-quality signals."
    )

    left, right = st.columns([0.62, 0.38])
    with left:
        video_url = st.text_input(
            "Video URL",
            placeholder="https://example.com/video.mp4 or a public video page supported by yt-dlp",
        )
        uploaded_video = st.file_uploader(
            "Or upload a video",
            type=["mp4", "mov", "m4v", "webm", "avi", "mkv"],
        )

        if uploaded_video is not None:
            st.video(uploaded_video)
        elif video_url:
            st.video(video_url)

    with right:
        sample_every = st.slider("Sample every N seconds", 1.0, 6.0, 2.0, 0.5)
        max_samples = st.slider("Max sampled frames", 40, 240, 160, 20)
        max_mb = st.slider("Max download size MB", 50, 600, 300, 50)
        include_audio = st.checkbox(
            "Analyze audio track",
            value=True,
            help="Adds silence, energy, audio-change, and peak-level signals to the retention risk score.",
        )
        include_transcript = st.checkbox(
            "Try transcript understanding",
            value=False,
            help=(
                "Optional and slower. Requires faster-whisper or openai-whisper installed locally. "
                "When available, speech content is scored for hook and clarity."
            ),
        )
        cookies_choice = st.selectbox(
            "Browser cookies for restricted URLs",
            ["None", "Chrome", "Safari", "Firefox", "Edge"],
            help=(
                "Use this only for videos you are allowed to access. It lets yt-dlp reuse "
                "your browser cookies when a platform blocks anonymous downloads."
            ),
        )
        st.write(
            "Lower intervals are more detailed but slower. For a first pass, 2 seconds is a good balance."
        )

    analyze_button = st.button("Analyze Video Retention", type="primary")

    if analyze_button:
        if uploaded_video is None and not video_url.strip():
            st.warning("Paste a video URL or upload a video first.")
            return

        tmp_path = None
        try:
            with st.spinner("Sampling frames and estimating drop-off risk"):
                if uploaded_video is not None:
                    suffix = Path(uploaded_video.name).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_video.getbuffer())
                        tmp_path = tmp.name
                    result = analyze_video_file(
                        tmp_path,
                        sample_every=sample_every,
                        max_samples=max_samples,
                        include_audio=include_audio,
                        include_transcript=include_transcript,
                    )
                else:
                    result = analyze_video_url(
                        video_url.strip(),
                        sample_every=sample_every,
                        max_samples=max_samples,
                        max_mb=max_mb,
                        cookies_browser=None if cookies_choice == "None" else cookies_choice.lower(),
                        include_audio=include_audio,
                        include_transcript=include_transcript,
                    )

            render_video_results(result)
        except Exception as exc:
            st.error(str(exc))
            st.info(
                "Some platforms block automated video downloads even when the video plays in your browser. "
                "The most reliable path is to upload the video file directly, use a public direct .mp4/.webm/.mov URL, "
                "or retry with browser cookies enabled for a video you are allowed to access."
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


st.title("Multimodal Engagement Studio")

video_tab, image_tab = st.tabs(["Video Retention Analyzer", "Image + Caption Predictor"])

with video_tab:
    render_video_retention_tab()

with image_tab:
    render_image_caption_tab()
