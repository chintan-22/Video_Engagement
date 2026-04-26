# Video Engagement and Retention Analyzer

An interview-ready multimodal ML prototype for predicting content engagement and identifying likely viewer drop-off moments in videos.

The project is built with Python and Streamlit. It includes two demos:

- **Video Retention Analyzer:** accepts a video URL or uploaded video, studies the video and audio, then highlights timestamps where viewers may lose interest.
- **Image + Caption Engagement Predictor:** accepts an image and caption, then predicts `High`, `Medium`, or `Low` engagement using fused visual and text features.

This is a prototype for demonstrating multimodal ML system design. It is not production-grade, and it is not a fully trained retention model with real platform watch-time labels.

## Core Idea

Modern engagement depends on more than one signal. A strong system should look at what viewers see, what they hear, and what the content says.

This project uses:

- **Video features:** frame sampling, pacing, visual change, brightness, sharpness, contrast, saturation, and static stretches.
- **Audio features:** silence, low energy, abrupt audio drops, flat audio patterns, weak opening audio hook, and possible clipping.
- **Text features:** Gemma-style scoring for hooks, emotion, clarity, curiosity, and call-to-action strength.
- **Optional transcript understanding:** if a local Whisper backend is installed, spoken content can be transcribed and scored.
- **Multimodal fusion:** combines multiple signal types into a timestamped risk estimate.

## What the App Produces

For a video, the Streamlit app shows:

- Predicted retention curve
- Drop-off risk timeline
- Visual, audio, and transcript risk breakdown
- Timestamped moments where viewers may lose interest
- Reasons for each risk spike
- Producer-facing recommendations for improving engagement

Example output style:

```text
Timestamp: 00:18
Risk: 74/100
Why: low visual change, flat audio pattern, weak hook
Fix: add a cutaway, tighten the spoken point, or move the payoff earlier
```

## Architecture

```text
                  Video URL / Upload
                         |
                  Download / Decode
                         |
        +----------------+----------------+
        |                                 |
   Frame Sampling                    Audio Extraction
        |                                 |
 Visual quality + pacing           Energy + silence
        |                                 |
        +----------------+----------------+
                         |
          Optional Transcript Understanding
                         |
          Multimodal timestamp-level scoring
                         |
       Retention risk curve + producer advice
```

The image-caption predictor uses a separate but related pipeline:

```text
Image -> CLIP image encoder -> image embedding
Caption -> Gemma-style scorer -> text quality scores
Image + text + metadata -> Random Forest classifier
Prediction -> High / Medium / Low engagement
```

## Project Structure

```text
Video_Engagement/
  README.md
  .gitignore
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
      predict.py
      train.py
      utils.py
      video_analysis.py
    notebooks/
      embedding_analysis.ipynb
    artifacts/
      .gitkeep
```

## Setup

From the repo root:

```bash
cd multimodal-engagement-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already created the virtual environment:

```bash
cd multimodal-engagement-predictor
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

If port `8501` is busy:

```bash
streamlit run app.py --server.port 8502
```

Then open the URL Streamlit prints, usually:

```text
http://localhost:8501
```

or:

```text
http://localhost:8502
```

## Using the Video Retention Analyzer

In the app, open the **Video Retention Analyzer** tab.

You can provide:

- A direct public video URL such as `.mp4`, `.webm`, or `.mov`
- A public video page URL supported by `yt-dlp`
- A local video upload

For best reliability, upload the video file directly. Some sites block automated downloads, require login, use DRM, or generate expiring URLs.

The app includes these controls:

- **Sample every N seconds:** controls timestamp granularity.
- **Analyze audio track:** includes audio energy and silence analysis.
- **Try transcript understanding:** uses local Whisper if installed.
- **Browser cookies for restricted URLs:** optional helper for URLs you are allowed to access.

## Using the Image + Caption Predictor

First train the lightweight classifier:

```bash
python src/train.py
```

Then run Streamlit:

```bash
streamlit run app.py
```

Open the **Image + Caption Predictor** tab, upload an image, enter a caption, and click **Predict Engagement**.

The model returns:

- Predicted class: `High`, `Medium`, or `Low`
- Confidence score
- Gemma-style caption scores
- Class probabilities
- Short explanation of contributing signals

## Dataset

The sample image-caption dataset lives at:

```text
multimodal-engagement-predictor/data/sample_data.csv
```

It contains synthetic rows with:

- `image_path`
- `caption`
- `likes`
- `engagement_label`

The project includes placeholder images so the demo can run immediately. For a stronger real-world demo, replace the sample data with real post images and engagement labels.

## Models and Fallbacks

The project is designed to run on CPU.

- CLIP is attempted through Hugging Face Transformers.
- If CLIP is unavailable, handcrafted image features are used.
- Gemma is attempted when enabled or locally cached.
- If Gemma is unavailable, deterministic text scoring is used.
- Audio extraction uses `imageio-ffmpeg` when system `ffmpeg` is not installed.
- Transcript scoring is optional and requires `faster-whisper` or `openai-whisper`.

To try Gemma locally:

```bash
export ENABLE_GEMMA=1
```

To add transcript understanding:

```bash
pip install faster-whisper
```

## Video Retention Connection

This prototype predicts engagement from image-caption pairs using multimodal embeddings. For a video-retention prediction system, this same idea can be extended by splitting a video into second-level chunks, extracting visual embeddings from frames, audio embeddings from spectrograms, and text embeddings from transcripts, then using a temporal model to predict retention at each timestamp.

## Limitations

- The retention analyzer estimates risk from content signals; it does not have real viewer watch-time labels.
- The image-caption classifier is trained on a tiny synthetic dataset.
- Gemma is used for prompt-style scoring, not as a fully fine-tuned engagement model.
- Some video URLs may fail because of platform restrictions, login requirements, DRM, or expiring links.
- Real engagement depends on distribution, creator history, audience, thumbnail, title, timing, and platform algorithms.

## Future Improvements

- Train on real timestamp-level retention curves.
- Add transcript embeddings and audio spectrogram embeddings.
- Use CLIP or video transformers on sampled frames.
- Add a temporal model such as LSTM, Transformer, or Temporal CNN.
- Add title, thumbnail, comments, creator metadata, and audience segment features.
- Calibrate predictions with real platform analytics.
- Export an edit decision list for producers.

## 30-Second Interview Pitch

I built a multimodal engagement prototype that analyzes both static image-caption posts and full videos. For video, it samples frames, extracts audio signals, optionally understands transcripts, and produces a timestamped retention-risk curve with producer recommendations. For image-caption posts, it combines CLIP-style visual features with Gemma-style text scoring and trains a lightweight classifier. The goal is not to claim production accuracy, but to show how I would design a practical multimodal ML system that connects model outputs to actionable content decisions.
