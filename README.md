# ML-audio-processing-app

Dash-based web app for recording, organizing, processing, and classifying audio samples.

## Features

- Studio tab for browser-based audio recording with the `S_<sample>_<g|b>` naming convention
- File Management tab for dataset bundle selection, folder-tree browsing, sub-folder creation, and uploads into:
  - `Train/Validation Pool`
  - `Testing`
- Root-level uploads can preserve folder structure, so you can import a folder like `Set1` directly or choose a parent folder containing multiple sets
- Processing tab for reusable saved-run loading, optional retraining, preprocessing, onset splitting, PSD/MFCC feature extraction, and running:
  - KNN
  - Decision Tree
  - Logistic Regression
  - SVM
- Results / Analysis tab for:
  - raw vs processed waveform previews
  - onset detection and hit-length inspection
  - confusion matrices
  - training/validation/testing accuracy comparison
  - robustness ranking and hyperparameter summaries

## Project Structure

- `app.py` - Dash entry point
- `ml_audio_app/layout.py` - tab layouts and main Dash component tree
- `ml_audio_app/callbacks.py` - app callbacks and processing workflow wiring
- `ml_audio_app/pipeline.py` - audio preprocessing, onset splitting, feature extraction, model training
- `ml_audio_app/storage.py` - file management and saved-run persistence
- `ml_audio_app/plots.py` - Plotly figure helpers
- `assets/` - custom CSS plus browser-side helpers for recording, uploads, and processing timer UI
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

## Workflow Overview

### Studio

- Browser recordings are saved as `.wav` for compatibility
- Recording names follow the assignment convention: `S_1_g.wav`, `S_12_b.wav`, etc.
- Recorded files are saved into the currently selected dataset bundle and folder target

### File Management

- Source audio is organized by dataset bundle, then by section:
  - `data/<dataset>/pooled/...`
  - `data/<dataset>/validation/...`
- `Train/Validation Pool` is the shared source used for runtime splitting
- `Testing` is reserved for independent held-out recordings
- The on-disk folder is still named `data/<dataset>/validation/...` for backward compatibility
- If a root section is selected, the upload area supports full-folder imports
- If a sub-folder is selected, uploaded files go directly into that folder

### Processing

- The app attaches the latest saved run for the active dataset automatically when available
- `Use an existing saved run` avoids rebuilding the pooled dataset and retraining models
- `Train new models` rebuilds features, refits classifiers, and saves a new `.joblib` artifact
- The advanced settings section contains the training/validation split ratio, onset-splitting parameters, and optional bandpass settings
- A live loader bar and elapsed timer appear while the processing pipeline is running

### Results / Analysis

- Signal preview uses the current preprocessing configuration
- Confusion matrices and result tables are populated from the active saved or freshly trained run
- Testing accuracy is shown alongside validation/training metrics for comparison

## Notes

- Uploaded and recorded source files live under dataset bundles like `data/workspace/...` or `data/example-hw3/...`
- The bundled demo dataset `Example - HW3` is included for end-to-end app demonstration
- Saved runs are written to `artifacts/runs`
- Optional split single-hit clips are written to `artifacts/processed_hits`
- The processing pipeline expects filenames ending in `_g` or `_b` so labels can be inferred automatically
- After changing browser-side assets in `assets/`, do a hard refresh if the app still appears to use older UI behavior
