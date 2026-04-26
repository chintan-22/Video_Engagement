# Multimodal Engagement Studio using Gemma + CLIP

An interview-ready Python prototype with two Streamlit demos:

1. Predict whether an image-caption social post is likely to receive **High**, **Medium**, or **Low** engagement.
2. Analyze a video URL or uploaded video and estimate where viewers may lose interest, with timestamped producer recommendations using both video and audio signals.

This is intentionally a small prototype, not a production-grade ranking system and not a fully fine-tuned Gemma model. The goal is to demonstrate multimodal ML thinking: visual features, text-side reasoning, feature fusion, a lightweight classifier, and an interactive demo.

## Why This Is Multimodal AI

The model uses both modalities instead of treating engagement as a text-only problem:

- **Image signal:** CLIP image embeddings summarize visual content.
- **Text signal:** Gemma-style caption scoring evaluates hook strength, emotional appeal, clarity, curiosity gap, and call-to-action strength.
- **Fusion:** The image embedding, caption scores, caption metadata, sentiment-like score, and compact text features are concatenated into one feature vector.
- **Classifier:** A Random Forest predicts `High`, `Medium`, or `Low` engagement.

## Architecture

```text
                 Image                         Caption
                   |                              |
        Hugging Face CLIP encoder        Gemma prompt-based scorer
                   |                              |
          512-d image embedding       hook / emotion / clarity / curiosity / CTA
                   |                              |
                   +------------+-----------------+
                                |
                 caption length + sentiment-like metadata
                                |
                         fused feature vector
                                |
                    lightweight Random Forest
                                |
                    High / Medium / Low engagement
```

Gemma can be heavy on a laptop. The code first supports Gemma scoring through Hugging Face Transformers, and falls back to deterministic heuristic scoring if Gemma is not locally available. CLIP also has a handcrafted image-feature fallback so the demo can still run on CPU without cached model weights.

## Video Retention Connection

“This prototype predicts engagement from image-caption pairs using multimodal embeddings. For a video-retention prediction system, this same idea can be extended by splitting a video into second-level chunks, extracting visual embeddings from frames, audio embeddings from spectrograms, and text embeddings from transcripts, then using a temporal model to predict retention at each timestamp.”

## Video Retention Analyzer

The Streamlit app includes a **Video Retention Analyzer** tab. It accepts:

- A direct public video URL such as `.mp4`, `.webm`, or `.mov`
- A public video page URL supported by `yt-dlp`
- A local video upload

The analyzer samples frames across the video and estimates drop-off risk using visual signals:

- Visual change and pacing
- Static stretches
- Sharpness / blur
- Brightness and contrast
- Saturation and visual detail
- Opening-hook strength in the first few seconds

It can also analyze the audio track:

- Silence or dead air
- Low audio energy
- Abrupt audio energy drops
- Flat/monotone audio patterns
- Possible clipping or harsh peaks
- Weak audio hook in the opening seconds

There is also an optional **Try transcript understanding** switch. If `faster-whisper` or `openai-whisper` is installed locally, the app can transcribe speech and score nearby transcript segments for hook strength and clarity. If no local transcription backend is available, the app still runs with visual + audio-energy analysis.

The output includes:

- Predicted retention curve
- Drop-off risk timeline
- Visual/audio/transcript risk breakdown
- Riskiest timestamps
- Reasons viewers may lose interest
- Producer-facing edit recommendations

This is a retention-risk prototype, not a substitute for actual platform watch-time analytics.

## Project Structure

```text
multimodal-engagement-predictor/
  README.md
  requirements.txt
  app.py
  data/
    sample_data.csv
    images/
  src/
    config.py
    feature_extraction.py
    train.py
    predict.py
    video_analysis.py
    utils.py
  notebooks/
    embedding_analysis.ipynb
  artifacts/
```

## Setup

```bash
cd multimodal-engagement-predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To enable real Gemma scoring, make sure you have access to the Gemma model on Hugging Face, then run:

```bash
export ENABLE_GEMMA=1
```

Without that flag, the project uses a fast fallback scorer so the prototype remains runnable on CPU.

## Dataset

The sample dataset is in `data/sample_data.csv` and contains 45 synthetic rows:

- `image_path`
- `caption`
- `likes`
- `engagement_label`

The CSV references image files under `data/images/`. If those files are missing, the training script creates deterministic placeholder images. For a stronger demo, replace the placeholder images with real social media images and keep the same paths in the CSV.

## Train

```bash
python src/train.py
```

The script:

- Loads `data/sample_data.csv`
- Extracts fused image + caption features
- Splits train/test data
- Trains a Random Forest classifier
- Prints accuracy and a classification report
- Saves the trained bundle to `artifacts/model.pkl`

## Run the Streamlit Demo

```bash
streamlit run app.py
```

In the app:

- Use **Video Retention Analyzer** to paste a video URL or upload a video, then click **Analyze Video Retention**.
- Use **Image + Caption Predictor** to upload an image, enter a caption, and click **Predict Engagement**.

For URL analysis, direct video links work best. YouTube and other page URLs require `yt-dlp`, which is included in `requirements.txt`.

## Embedding Analysis

Open `notebooks/embedding_analysis.ipynb` after training. It loads fused embeddings, applies PCA, and plots samples colored by engagement label. This helps inspect whether high-engagement examples cluster separately from medium and low examples.

## CPU-Friendly Behavior

This repo is designed to run on CPU:

- CLIP is attempted through Hugging Face Transformers.
- If CLIP is unavailable, handcrafted image features are used.
- Gemma is attempted when enabled or locally cached.
- If Gemma is unavailable, deterministic text scoring is used.

This keeps the prototype reliable during an interview while still showing where CLIP and Gemma fit in the architecture.

## Limitations

- The dataset is synthetic and tiny, so reported metrics are demonstration-only.
- Gemma is used for prompt-based caption scoring, not fine-tuned engagement prediction.
- Placeholder images are useful for pipeline testing but not meaningful for real-world model quality.
- Engagement labels are simplified to three classes.
- Real engagement is affected by creator history, platform algorithms, timing, audience, and distribution effects.
- The video analyzer estimates retention risk from content signals; true retention prediction requires historical watch-time labels.
- Some video URLs may fail because of authentication, DRM, platform restrictions, or unsupported formats.
- Transcript understanding is optional and requires a local Whisper backend; otherwise the app uses audio-energy analysis only.

## Future Improvements

- Train on real image-caption engagement data.
- Train the video-retention model on real timestamp-level retention curves.
- Add transcript analysis and audio-energy features.
- Add multimodal temporal modeling across frame, audio, and transcript embeddings.
- Add calibrated probabilities and cross-validation.
- Fine-tune a text encoder or use Gemma embeddings through a hosted inference path.
- Add richer image quality features such as aesthetics, faces, OCR, and composition.
- Add temporal modeling for carousel posts or short-form videos.
- Track experiments with MLflow or Weights & Biases.

## 30-Second Interview Pitch

I built a compact multimodal engagement predictor that takes both an image and a caption. CLIP extracts the visual representation, Gemma-style scoring evaluates the caption for hook, emotion, clarity, curiosity, and call-to-action, and those features are fused with simple metadata before training a lightweight Random Forest. It is not meant to be production-grade; it is a prototype showing how I would structure a multimodal ML system, build fallbacks for practical constraints, and expose the result through a Streamlit demo.
