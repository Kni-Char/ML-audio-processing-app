from __future__ import annotations

from .pipeline import ExperimentConfig, PreprocessingConfig, SplitConfig


def parse_delta_list(raw_value: str) -> list[float]:
    raw_value = raw_value or ""
    if not raw_value.strip():
        return [0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04]
    values = []
    for token in raw_value.split(","):
        values.append(float(token.strip()))
    return values


def build_experiment_config(
    preprocessing_flags: list[str],
    lowcut_hz: float,
    highcut_hz: float,
    filter_order: int,
    pre_sec: float,
    post_sec: float,
    min_gap_sec: float,
    hop_length: int,
    delta_list: str,
    bad_hits: int,
    selected_models: list[str],
    selected_feature_sets: list[str],
    save_processed_hits_flag: list[str],
) -> ExperimentConfig:
    preprocessing = PreprocessingConfig(
        enabled="enabled" in preprocessing_flags,
        remove_dc="remove_dc" in preprocessing_flags,
        normalize_peak="normalize_peak" in preprocessing_flags,
        apply_bandpass="apply_bandpass" in preprocessing_flags,
        lowcut_hz=float(lowcut_hz),
        highcut_hz=float(highcut_hz),
        filter_order=int(filter_order),
    )
    split = SplitConfig(
        pre_sec=float(pre_sec),
        post_sec=float(post_sec),
        min_gap_sec=float(min_gap_sec),
        hop_length=int(hop_length),
        delta_list=parse_delta_list(delta_list),
    )
    return ExperimentConfig(
        preprocessing=preprocessing,
        split=split,
        bad_hits=int(bad_hits),
        selected_models=list(selected_models or []),
        selected_feature_sets=list(selected_feature_sets or []),
        save_processed_hits="save" in (save_processed_hits_flag or []),
    )
