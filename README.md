# ML-audio-processing-app

Dash-based web app for recording, organizing, processing, and classifying audio samples.

## Features

- Studio tab for browser-based audio recording with the `S_<sample>_<g|b>` naming convention
- File Management tab for drag/drop uploads, dataset bundle selection, and assignment into `Training`, `Testing`, and `Validation`
- Processing tab for safe preprocessing, onset splitting, PSD/MFCC feature extraction, and running:
  - KNN
  - Decision Tree
  - Logistic Regression
  - SVM
- Results / Analysis tab for:
  - raw vs processed waveform previews
  - onset detection and hit-length inspection
  - confusion matrices
  - training/testing/validation accuracy comparison
  - robustness ranking and hyperparameter summaries

## Project Structure

- `app.py` - Dash entry point
- `ml_audio_app/pipeline.py` - audio preprocessing, onset splitting, feature extraction, model training
- `ml_audio_app/storage.py` - file management and saved-run persistence
- `ml_audio_app/plots.py` - Plotly figure helpers
- `assets/` - custom CSS and browser recording JavaScript
- `data/<dataset>/<section>/<group>/...` - dataset bundles with nested groups such as `Set1` or `Independent-Day`

## Run Locally

1. Install dependencies with either `pip` or `uv`.

```powershell
pip install -r requirements.txt
```

Or with `uv`:

```powershell
uv sync
```

2. Start the app:

```powershell
python app.py
```

Or with `uv`:

```powershell
uv run app.py
```

3. Open the local Dash URL shown in the terminal.

## Notes

- Uploaded and recorded source files live under dataset bundles like `data/workspace/...` or `data/example-hw3/...`.
- The bundled demo dataset `Example - HW3` assumes `Set1`-`Set4` are training, `Set5` is testing, and `Validation_Data` is the independent validation set.
- Saved runs are written to `artifacts/runs`.
- Optional split single-hit clips are written to `artifacts/processed_hits`.
- The processing pipeline expects filenames ending in `_g` or `_b` so labels can be inferred automatically.
