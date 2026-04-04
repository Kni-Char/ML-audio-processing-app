from __future__ import annotations

import io
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks, stft as scipy_stft, welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import DEFAULT_BAD_HITS, GOOD_HITS, PROCESSED_ROOT, SECTION_NAMES
from .storage import get_dataset_label, list_audio_paths, section_dir


N_MFCC = 20
PSD_BINS = 40

MODEL_ORDER = ("KNN", "Decision Tree", "Logistic Regression", "SVM")
FEATURE_SET_ORDER = ("PSD", "MFCC")
LABELS_DISPLAY = ["good (0)", "bad (1)"]


@dataclass
class PreprocessingConfig:
    enabled: bool = True
    remove_dc: bool = True
    normalize_peak: bool = True
    apply_bandpass: bool = True
    lowcut_hz: float = 80.0
    highcut_hz: float = 8000.0
    filter_order: int = 4


@dataclass
class SplitConfig:
    pre_sec: float = 0.02
    post_sec: float = 0.20
    min_gap_sec: float = 0.04
    hop_length: int = 256
    delta_list: list[float] = field(default_factory=lambda: [0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04])


@dataclass
class ExperimentConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    bad_hits: int = DEFAULT_BAD_HITS
    selected_models: list[str] = field(default_factory=lambda: list(MODEL_ORDER))
    selected_feature_sets: list[str] = field(default_factory=lambda: list(FEATURE_SET_ORDER))
    save_processed_hits: bool = True


@dataclass
class ProcessedAudioFile:
    raw_signal: np.ndarray
    processed_signal: np.ndarray
    sample_rate: int
    onset_frames: np.ndarray
    onset_envelope: np.ndarray
    used_delta: float | None
    clips: list[np.ndarray]
    expected_hits: int
    label: int


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def experiment_config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        preprocessing=PreprocessingConfig(**data["preprocessing"]),
        split=SplitConfig(**data["split"]),
        bad_hits=int(data["bad_hits"]),
        selected_models=list(data["selected_models"]),
        selected_feature_sets=list(data["selected_feature_sets"]),
        save_processed_hits=bool(data.get("save_processed_hits", True)),
    )


def safe_preprocess_waveform(y: np.ndarray, sr: int, config: PreprocessingConfig) -> np.ndarray:
    y_proc = y.astype(np.float32).copy()

    if not config.enabled:
        return y_proc

    if config.remove_dc:
        y_proc = y_proc - np.mean(y_proc)

    if config.apply_bandpass:
        nyquist = 0.5 * sr
        low = max(1e-6, config.lowcut_hz / nyquist)
        high = min(0.999999, config.highcut_hz / nyquist)
        if low < high:
            b, a = butter(int(config.filter_order), [low, high], btype="band")
            y_proc = filtfilt(b, a, y_proc)

    if config.normalize_peak:
        peak = np.max(np.abs(y_proc))
        if peak > 1e-12:
            y_proc = 0.99 * (y_proc / peak)

    return y_proc


def label_from_name(filename: str) -> int:
    stem = Path(filename).stem.lower()
    if stem.endswith("_g"):
        return 0
    if stem.endswith("_b"):
        return 1
    raise ValueError(f"Cannot infer label from filename '{filename}'. Expected a suffix of '_g' or '_b'.")


def expected_hits_from_name(filename: str, bad_hits: int) -> int:
    return GOOD_HITS if label_from_name(filename) == 0 else bad_hits


def enforce_min_gap(frames: np.ndarray, min_gap_frames: int) -> np.ndarray:
    frames = np.sort(np.array(frames, dtype=int))
    if len(frames) == 0:
        return frames

    kept = [int(frames[0])]
    for frame in frames[1:]:
        if frame - kept[-1] >= min_gap_frames:
            kept.append(int(frame))
    return np.array(kept, dtype=int)


def strongest_k(frames: np.ndarray, onset_envelope: np.ndarray, k: int) -> np.ndarray:
    if len(frames) <= k:
        return np.array(frames, dtype=int)
    strengths = onset_envelope[frames]
    keep = np.argsort(strengths)[::-1][:k]
    return np.sort(np.array(frames, dtype=int)[keep])


def detect_onsets_adaptive(
    y: np.ndarray,
    sr: int,
    expected: int,
    split_config: SplitConfig,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    onset_envelope = compute_onset_envelope(y, split_config.hop_length)
    min_gap_frames = max(1, int((split_config.min_gap_sec * sr) / split_config.hop_length))
    best_frames = np.array([], dtype=int)
    best_delta = None
    envelope_range = max(float(np.max(onset_envelope) - np.min(onset_envelope)), 1e-6)
    baseline = float(np.median(onset_envelope))

    for delta in split_config.delta_list:
        threshold = baseline + delta * envelope_range
        prominence = max(1e-6, delta * envelope_range * 0.35)
        frames, _ = find_peaks(
            onset_envelope,
            height=threshold,
            prominence=prominence,
            distance=min_gap_frames,
        )
        frames = enforce_min_gap(frames, min_gap_frames)
        if len(frames) > expected:
            frames = strongest_k(frames, onset_envelope, expected)

        if len(frames) > len(best_frames):
            best_frames = frames
            best_delta = delta

        if len(frames) == expected:
            return frames, onset_envelope, delta

    return best_frames, onset_envelope, best_delta


def compute_onset_envelope(y: np.ndarray, hop_length: int, frame_length: int = 1024) -> np.ndarray:
    magnitude = np.abs(y).astype(np.float32)
    window = np.ones(frame_length, dtype=np.float32) / frame_length
    smoothed = np.convolve(magnitude, window, mode="same")
    envelope = smoothed[::hop_length]
    if envelope.size == 0:
        return np.array([0.0], dtype=np.float32)
    return envelope.astype(np.float32)


def split_hits(
    y: np.ndarray,
    sr: int,
    onset_frames: np.ndarray,
    split_config: SplitConfig,
) -> list[np.ndarray]:
    onset_samples = onset_frames * split_config.hop_length
    clips: list[np.ndarray] = []
    pre = int(split_config.pre_sec * sr)
    post = int(split_config.post_sec * sr)

    for sample in onset_samples:
        start = max(0, int(sample) - pre)
        end = min(len(y), int(sample) + post)
        clip = y[start:end]
        if len(clip) > int(0.03 * sr):
            clips.append(clip)

    return clips


def process_audio_file(
    file_path: Path,
    preprocessing: PreprocessingConfig,
    split_config: SplitConfig,
    bad_hits: int,
) -> ProcessedAudioFile:
    y_raw, sr = load_audio_mono(file_path)
    y_processed = safe_preprocess_waveform(y_raw, sr, preprocessing)
    expected_hits = expected_hits_from_name(file_path.name, bad_hits)
    onset_frames, onset_envelope, used_delta = detect_onsets_adaptive(
        y_processed,
        sr,
        expected_hits,
        split_config,
    )
    clips = split_hits(y_processed, sr, onset_frames, split_config)
    label = label_from_name(file_path.name)
    return ProcessedAudioFile(
        raw_signal=y_raw,
        processed_signal=y_processed,
        sample_rate=sr,
        onset_frames=onset_frames,
        onset_envelope=onset_envelope,
        used_delta=used_delta,
        clips=clips,
        expected_hits=expected_hits,
        label=label,
    )


def load_audio_mono(file_path: Path) -> tuple[np.ndarray, int]:
    suffix = file_path.suffix.lower()
    if suffix in {".m4a", ".mp3", ".webm"}:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_exe,
            "-v",
            "error",
            "-i",
            str(file_path),
            "-vn",
            "-ac",
            "1",
            "-f",
            "wav",
            "-",
        ]
        result = subprocess.run(command, capture_output=True, check=True)
        audio_bytes = io.BytesIO(result.stdout)
        y, sr = sf.read(audio_bytes, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y.astype(np.float32), int(sr)

    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y.astype(np.float32), int(sr)


def extract_mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def extract_psd_features(y: np.ndarray, sr: int, psd_bins: int = PSD_BINS) -> np.ndarray:
    _, pxx = welch(y, fs=sr, nperseg=min(1024, len(y)))
    pxx = np.log10(pxx + 1e-12)
    if len(pxx) >= psd_bins:
        return pxx[:psd_bins]
    return np.pad(pxx, (0, psd_bins - len(pxx)), mode="constant")


def build_feature_matrix(clips: list[tuple[np.ndarray, int]], feature_name: str) -> np.ndarray:
    rows: list[np.ndarray] = []
    for clip, sr in clips:
        if feature_name == "PSD":
            rows.append(extract_psd_features(clip, sr))
        elif feature_name == "MFCC":
            rows.append(extract_mfcc_features(clip, sr))
        else:
            raise ValueError(f"Unknown feature set: {feature_name}")

    if not rows:
        width = PSD_BINS if feature_name == "PSD" else N_MFCC * 2
        return np.empty((0, width), dtype=float)

    return np.vstack(rows)


def build_models(selected_models: list[str]) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    models = {
        "KNN": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            {
                "model__n_neighbors": [1, 3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [None, 3, 5, 7, 10, 15],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "Logistic Regression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=5000, random_state=42)),
                ]
            ),
            {
                "model__C": [0.01, 0.1, 1, 10, 100],
            },
        ),
        "SVM": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVC(random_state=42)),
                ]
            ),
            {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
            },
        ),
    }
    return {name: models[name] for name in MODEL_ORDER if name in selected_models}


def build_run_id(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def _file_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        columns=[
            "section",
            "group",
            "relative_path",
            "source_file",
            "hit_index",
            "label",
            "status",
            "error",
        ]
    )


def process_section_dataset(
    section: str,
    config: ExperimentConfig,
    processed_root: Path | None,
    dataset_slug: str,
) -> dict[str, Any]:
    folder = section_dir(dataset_slug, section)
    files = list_audio_paths(dataset_slug, section)

    clips_all: list[tuple[np.ndarray, int]] = []
    labels_all: list[int] = []
    meta_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    group_names: set[str] = set()

    output_dir = None
    if processed_root is not None:
        output_dir = processed_root / section
        output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        relative_path = file_path.relative_to(folder)
        group_key = "" if relative_path.parent == Path(".") else relative_path.parent.as_posix()
        group_label = group_key or "Root"
        group_names.add(group_label)
        try:
            processed = process_audio_file(
                file_path=file_path,
                preprocessing=config.preprocessing,
                split_config=config.split,
                bad_hits=config.bad_hits,
            )

            for index, clip in enumerate(processed.clips):
                clips_all.append((clip, processed.sample_rate))
                labels_all.append(processed.label)
                meta_rows.append(
                    {
                        "section": section,
                        "group": group_label,
                        "relative_path": relative_path.as_posix(),
                        "source_file": file_path.name,
                        "hit_index": index,
                        "label": processed.label,
                        "status": "ok",
                        "error": "",
                    }
                )
                if output_dir is not None:
                    clip_output_dir = output_dir / Path(group_key) if group_key else output_dir
                    clip_output_dir.mkdir(parents=True, exist_ok=True)
                    sf.write(clip_output_dir / f"{file_path.stem}_hit{index:02d}.wav", clip, processed.sample_rate)

            diag_rows.append(
                {
                    "section": section,
                    "group": group_label,
                    "relative_path": relative_path.as_posix(),
                    "source_file": file_path.name,
                    "status": "ok",
                    "found_hits": len(processed.clips),
                    "expected_hits": processed.expected_hits,
                    "delta_used": processed.used_delta,
                    "duration_sec": round(len(processed.raw_signal) / processed.sample_rate, 3),
                    "error": "",
                }
            )
        except Exception as exc:
            diag_rows.append(
                {
                    "section": section,
                    "group": group_label,
                    "relative_path": relative_path.as_posix(),
                    "source_file": file_path.name,
                    "status": "error",
                    "found_hits": 0,
                    "expected_hits": "",
                    "delta_used": "",
                    "duration_sec": "",
                    "error": str(exc),
                }
            )

    meta_df = _file_rows_to_frame(meta_rows)
    diag_df = pd.DataFrame(diag_rows)
    labels = np.array(labels_all, dtype=int) if labels_all else np.array([], dtype=int)

    return {
        "clips": clips_all,
        "labels": labels,
        "meta_df": meta_df,
        "diag_df": diag_df,
        "summary": {
            "files": len(files),
            "groups": len(group_names),
            "clips": int(len(labels_all)),
            "good_clips": int(np.sum(labels == 0)),
            "bad_clips": int(np.sum(labels == 1)),
        },
    }


def prepare_datasets(config: ExperimentConfig, run_id: str, dataset_slug: str) -> dict[str, Any]:
    processed_root = PROCESSED_ROOT / run_id if config.save_processed_hits else None
    if processed_root is not None:
        processed_root.mkdir(parents=True, exist_ok=True)

    sections: dict[str, dict[str, Any]] = {}
    for section in SECTION_NAMES:
        sections[section] = process_section_dataset(section, config, processed_root, dataset_slug)
        if sections[section]["summary"]["clips"] == 0:
            raise ValueError(
                f"The '{section}' section did not produce any single-hit clips. "
                "Check that files follow the S_1_g / S_1_b naming convention and contain audible impacts."
            )

    for section in SECTION_NAMES:
        unique_classes = np.unique(sections[section]["labels"])
        if len(unique_classes) < 2:
            raise ValueError(
                f"The '{section}' section needs at least one good and one bad recording after splitting."
            )

    return sections


def _min_class_count(labels: np.ndarray) -> int:
    values, counts = np.unique(labels, return_counts=True)
    if len(values) < 2:
        return 0
    return int(np.min(counts))


def build_feature_sets(sections: dict[str, dict[str, Any]], feature_sets: list[str]) -> dict[str, dict[str, np.ndarray]]:
    matrices: dict[str, dict[str, np.ndarray]] = {section: {} for section in SECTION_NAMES}
    for section in SECTION_NAMES:
        for feature_name in feature_sets:
            matrices[section][feature_name] = build_feature_matrix(sections[section]["clips"], feature_name)
    return matrices


def _results_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_set",
                "model",
                "train_accuracy",
                "testing_accuracy",
                "validation_accuracy",
                "testing_to_validation_drop",
                "best_params",
            ]
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        by=["validation_accuracy", "testing_accuracy", "train_accuracy"],
        ascending=False,
    ).reset_index(drop=True)


def train_experiment(config: ExperimentConfig, dataset_slug: str) -> dict[str, Any]:
    if not config.selected_models:
        raise ValueError("Select at least one model before running the pipeline.")
    if not config.selected_feature_sets:
        raise ValueError("Select at least one feature set before running the pipeline.")

    run_id = build_run_id(f"run_{dataset_slug}")
    sections = prepare_datasets(config, run_id, dataset_slug)
    matrices = build_feature_sets(sections, config.selected_feature_sets)

    y_train = sections["training"]["labels"]
    y_test = sections["testing"]["labels"]
    y_validation = sections["validation"]["labels"]

    cv_splits = min(5, _min_class_count(y_train))
    if cv_splits < 2:
        raise ValueError("Training clips need at least two samples per class for cross-validation.")

    models = build_models(config.selected_models)
    results_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    confusion_payload: dict[str, dict[str, dict[str, list[list[int]]]]] = {}
    trained_models: dict[str, dict[str, Any]] = {}

    for feature_name in config.selected_feature_sets:
        confusion_payload[feature_name] = {}
        trained_models[feature_name] = {}
        X_train = matrices["training"][feature_name]
        X_test = matrices["testing"][feature_name]
        X_validation = matrices["validation"][feature_name]

        for model_name, (estimator, param_grid) in models.items():
            search = GridSearchCV(
                estimator,
                param_grid,
                cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42),
                n_jobs=1,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            yhat_train = best_model.predict(X_train)
            yhat_test = best_model.predict(X_test)
            yhat_validation = best_model.predict(X_validation)

            acc_train = float(accuracy_score(y_train, yhat_train))
            acc_test = float(accuracy_score(y_test, yhat_test))
            acc_validation = float(accuracy_score(y_validation, yhat_validation))

            best_params = search.best_params_
            results_rows.append(
                {
                    "feature_set": feature_name,
                    "model": model_name,
                    "train_accuracy": acc_train,
                    "testing_accuracy": acc_test,
                    "validation_accuracy": acc_validation,
                    "testing_to_validation_drop": acc_test - acc_validation,
                    "best_params": json.dumps(best_params, sort_keys=True),
                    "mode": "trained",
                }
            )

            trained_models[feature_name][model_name] = best_model
            confusion_payload[feature_name][model_name] = {
                "train": confusion_matrix(y_train, yhat_train).tolist(),
                "testing": confusion_matrix(y_test, yhat_test).tolist(),
                "validation": confusion_matrix(y_validation, yhat_validation).tolist(),
            }

            cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(5)
            for _, row in cv_results.iterrows():
                tuning_rows.append(
                    {
                        "feature_set": feature_name,
                        "model": model_name,
                        "rank": int(row["rank_test_score"]),
                        "mean_cv_score": round(float(row["mean_test_score"]), 4),
                        "std_cv_score": round(float(row["std_test_score"]), 4),
                        "params": json.dumps(row["params"], sort_keys=True),
                    }
                )

    results_df = _results_dataframe(results_rows)
    comparison_table = results_df.copy()
    for column in ["train_accuracy", "testing_accuracy", "validation_accuracy", "testing_to_validation_drop"]:
        comparison_table[column] = comparison_table[column].round(4)

    robustness_table = comparison_table.sort_values(
        by=["testing_to_validation_drop", "validation_accuracy"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return {
        "run_id": run_id,
        "mode": "trained",
        "dataset_slug": dataset_slug,
        "dataset_label": get_dataset_label(dataset_slug),
        "config": experiment_config_to_dict(config),
        "results_table": comparison_table.to_dict("records"),
        "robustness_table": robustness_table.to_dict("records"),
        "tuning_table": pd.DataFrame(tuning_rows).to_dict("records"),
        "confusion_matrices": confusion_payload,
        "trained_models": trained_models,
        "section_diagnostics": {
            section: sections[section]["diag_df"].to_dict("records") for section in SECTION_NAMES
        },
        "section_meta": {
            section: sections[section]["meta_df"].to_dict("records") for section in SECTION_NAMES
        },
        "section_summary": {
            section: sections[section]["summary"] for section in SECTION_NAMES
        },
    }


def evaluate_saved_experiment(saved_bundle: dict[str, Any], dataset_slug: str) -> dict[str, Any]:
    if "trained_models" not in saved_bundle:
        raise ValueError("The selected artifact does not contain trained models.")

    config = experiment_config_from_dict(saved_bundle["config"])
    run_id = build_run_id(f"eval_{dataset_slug}")
    sections = prepare_datasets(config, run_id, dataset_slug)
    matrices = build_feature_sets(sections, config.selected_feature_sets)

    y_train = sections["training"]["labels"]
    y_test = sections["testing"]["labels"]
    y_validation = sections["validation"]["labels"]

    results_rows: list[dict[str, Any]] = []
    confusion_payload: dict[str, dict[str, dict[str, list[list[int]]]]] = {}
    prior_results = {
        (row["feature_set"], row["model"]): row.get("best_params", "{}")
        for row in saved_bundle.get("results_table", [])
    }

    for feature_name in config.selected_feature_sets:
        confusion_payload[feature_name] = {}
        X_train = matrices["training"][feature_name]
        X_test = matrices["testing"][feature_name]
        X_validation = matrices["validation"][feature_name]

        for model_name in config.selected_models:
            estimator = saved_bundle["trained_models"][feature_name][model_name]

            yhat_train = estimator.predict(X_train)
            yhat_test = estimator.predict(X_test)
            yhat_validation = estimator.predict(X_validation)

            acc_train = float(accuracy_score(y_train, yhat_train))
            acc_test = float(accuracy_score(y_test, yhat_test))
            acc_validation = float(accuracy_score(y_validation, yhat_validation))

            results_rows.append(
                {
                    "feature_set": feature_name,
                    "model": model_name,
                    "train_accuracy": acc_train,
                    "testing_accuracy": acc_test,
                    "validation_accuracy": acc_validation,
                    "testing_to_validation_drop": acc_test - acc_validation,
                    "best_params": prior_results.get((feature_name, model_name), "{}"),
                    "mode": "loaded-evaluation",
                }
            )

            confusion_payload[feature_name][model_name] = {
                "train": confusion_matrix(y_train, yhat_train).tolist(),
                "testing": confusion_matrix(y_test, yhat_test).tolist(),
                "validation": confusion_matrix(y_validation, yhat_validation).tolist(),
            }

    results_df = _results_dataframe(results_rows)
    comparison_table = results_df.copy()
    for column in ["train_accuracy", "testing_accuracy", "validation_accuracy", "testing_to_validation_drop"]:
        comparison_table[column] = comparison_table[column].round(4)

    robustness_table = comparison_table.sort_values(
        by=["testing_to_validation_drop", "validation_accuracy"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return {
        "run_id": run_id,
        "mode": "loaded-evaluation",
        "source_run_id": saved_bundle.get("run_id"),
        "dataset_slug": dataset_slug,
        "dataset_label": get_dataset_label(dataset_slug),
        "config": experiment_config_to_dict(config),
        "results_table": comparison_table.to_dict("records"),
        "robustness_table": robustness_table.to_dict("records"),
        "tuning_table": saved_bundle.get("tuning_table", []),
        "confusion_matrices": confusion_payload,
        "trained_models": saved_bundle["trained_models"],
        "section_diagnostics": {
            section: sections[section]["diag_df"].to_dict("records") for section in SECTION_NAMES
        },
        "section_meta": {
            section: sections[section]["meta_df"].to_dict("records") for section in SECTION_NAMES
        },
        "section_summary": {
            section: sections[section]["summary"] for section in SECTION_NAMES
        },
    }


def build_signal_preview(
    file_path: Path,
    preprocessing: PreprocessingConfig,
    split_config: SplitConfig,
    bad_hits: int,
) -> dict[str, Any]:
    processed = process_audio_file(file_path, preprocessing, split_config, bad_hits)
    sr = processed.sample_rate
    time_raw = np.arange(len(processed.raw_signal)) / sr
    time_processed = np.arange(len(processed.processed_signal)) / sr
    onset_times = (processed.onset_frames * split_config.hop_length) / sr
    onset_time_axis = (np.arange(len(processed.onset_envelope)) * split_config.hop_length) / sr

    frequency_axis, spectrogram_times, zxx = scipy_stft(
        processed.processed_signal,
        fs=sr,
        nperseg=1024,
        noverlap=max(0, 1024 - split_config.hop_length),
        boundary=None,
    )
    magnitude = np.abs(zxx)
    spectrogram_db = 20.0 * np.log10(np.maximum(magnitude, 1e-10) / max(float(np.max(magnitude)), 1e-10))

    clip_lengths = [round(len(clip) / sr, 4) for clip in processed.clips]

    return {
        "file_name": file_path.name,
        "label": processed.label,
        "expected_hits": processed.expected_hits,
        "found_hits": len(processed.clips),
        "delta_used": processed.used_delta,
        "sample_rate": sr,
        "duration_sec": round(len(processed.raw_signal) / sr, 3),
        "raw_time": time_raw.tolist(),
        "raw_signal": processed.raw_signal.tolist(),
        "processed_time": time_processed.tolist(),
        "processed_signal": processed.processed_signal.tolist(),
        "onset_times": onset_times.tolist(),
        "onset_time_axis": onset_time_axis.tolist(),
        "onset_envelope": processed.onset_envelope.tolist(),
        "spectrogram_db": spectrogram_db.tolist(),
        "spectrogram_freqs": frequency_axis.tolist(),
        "spectrogram_times": spectrogram_times.tolist(),
        "clip_lengths": clip_lengths,
    }
