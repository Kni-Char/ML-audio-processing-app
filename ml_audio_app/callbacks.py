from __future__ import annotations

from dash import Input, Output, State, callback_context, no_update

from .config import DEFAULT_RECORDING_PREFIX, SECTION_LABELS
from .pipeline import build_signal_preview, evaluate_saved_experiment, train_experiment
from .plots import (
    make_accuracy_figure,
    make_clip_lengths_figure,
    make_onset_figure,
    make_spectrogram_figure,
    make_waveform_figure,
)
from .storage import (
    build_recording_name,
    delete_files,
    get_dataset_record,
    list_audio_files,
    list_group_summary_rows,
    list_run_artifacts,
    load_run_artifact,
    move_files,
    resolve_audio_path,
    save_recording,
    save_run_artifact,
    save_uploaded_files,
)
from .ui_helpers import (
    build_confusion_grid,
    build_dataset_banner,
    build_file_summary_cards,
    build_run_banner,
    flatten_diagnostics,
    make_columns,
    make_media_url,
    message_block,
)
from .workflow import build_experiment_config


def register_callbacks(app, default_active_dataset: str) -> None:
    @app.callback(
        Output("recording-filename-preview", "children"),
        Input("active-dataset-dropdown", "value"),
        Input("record-target-section", "value"),
        Input("record-group", "value"),
        Input("record-prefix", "value"),
        Input("record-sample-number", "value"),
        Input("record-condition", "value"),
    )
    def update_recording_filename(
        dataset_slug: str,
        section: str,
        group: str,
        prefix: str,
        sample_number: int,
        condition: str,
    ):
        name = build_recording_name(
            prefix=prefix or DEFAULT_RECORDING_PREFIX,
            sample_number=sample_number or 1,
            condition=condition or "g",
        )
        dataset_label = get_dataset_record(dataset_slug or default_active_dataset)["label"]
        section_label = SECTION_LABELS.get(section or "training", "Training")
        group_label = f" / {group}" if group else ""
        return f"Next saved recording: {dataset_label} / {section_label}{group_label} / {name}"

    @app.callback(
        Output("studio-message", "children"),
        Output("file-table", "data", allow_duplicate=True),
        Output("dataset-summary-cards", "children", allow_duplicate=True),
        Output("group-summary-table", "data", allow_duplicate=True),
        Output("active-dataset-banner", "children", allow_duplicate=True),
        Input("save-recording-btn", "n_clicks"),
        State("recording-data", "value"),
        State("active-dataset-dropdown", "value"),
        State("record-target-section", "value"),
        State("record-group", "value"),
        State("record-prefix", "value"),
        State("record-sample-number", "value"),
        State("record-condition", "value"),
        prevent_initial_call=True,
    )
    def persist_recording(
        _n_clicks: int,
        recording_data: str,
        dataset_slug: str,
        section: str,
        group: str,
        prefix: str,
        sample_number: int,
        condition: str,
    ):
        try:
            destination = save_recording(
                data_url=recording_data,
                dataset_slug=dataset_slug,
                section=section,
                group=group or "",
                prefix=prefix or DEFAULT_RECORDING_PREFIX,
                sample_number=int(sample_number or 1),
                condition=condition or "g",
            )
            rows = list_audio_files(dataset_slug)
            return (
                message_block(f"Saved recording to {get_dataset_record(dataset_slug)['label']} / {SECTION_LABELS[section]} as {destination.name}.", "success"),
                rows,
                build_file_summary_cards(rows),
                list_group_summary_rows(dataset_slug),
                build_dataset_banner(dataset_slug),
            )
        except Exception as exc:
            rows = list_audio_files(dataset_slug or default_active_dataset)
            active_slug = dataset_slug or default_active_dataset
            return (
                message_block(str(exc), "danger"),
                rows,
                build_file_summary_cards(rows),
                list_group_summary_rows(active_slug),
                build_dataset_banner(active_slug),
            )

    @app.callback(
        Output("file-table", "data", allow_duplicate=True),
        Output("file-table", "selected_rows"),
        Output("file-management-message", "children"),
        Output("dataset-summary-cards", "children", allow_duplicate=True),
        Output("group-summary-table", "data"),
        Output("group-summary-table", "columns"),
        Output("active-dataset-banner", "children", allow_duplicate=True),
        Input("active-dataset-dropdown", "value"),
        Input("file-upload", "contents"),
        Input("refresh-files-btn", "n_clicks"),
        Input("move-files-btn", "n_clicks"),
        Input("delete-files-btn", "n_clicks"),
        State("file-upload", "filename"),
        State("upload-section", "value"),
        State("upload-group", "value"),
        State("move-target-section", "value"),
        State("move-target-group", "value"),
        State("file-table", "data"),
        State("file-table", "derived_virtual_data"),
        State("file-table", "selected_rows"),
        prevent_initial_call=True,
    )
    def handle_file_actions(
        active_dataset_slug,
        upload_contents,
        _refresh_clicks,
        _move_files,
        _delete_clicks,
        upload_filenames,
        upload_section,
        upload_group,
        move_target_section,
        move_target_group,
        table_data,
        visible_rows,
        selected_rows,
    ):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
        dataset_slug = active_dataset_slug or default_active_dataset
        dataset_label = get_dataset_record(dataset_slug)["label"]
        message = message_block(f"{dataset_label} is ready for uploads and section moves.", "info")
        current_rows = visible_rows or table_data or list_audio_files(dataset_slug)
        selected_rows = selected_rows or []
        selected_ids = [current_rows[index]["file_id"] for index in selected_rows if index < len(current_rows)]

        try:
            if trigger == "active-dataset-dropdown":
                message = message_block(f"Showing files from {dataset_label}.", "info")
            if trigger == "file-upload" and upload_contents:
                result = save_uploaded_files(upload_contents, upload_filenames, dataset_slug, upload_section, upload_group or "")
                saved_count = len(result["saved"])
                skipped_count = len(result["skipped"])
                message_text = f"Uploaded {saved_count} file(s) to {dataset_label} / {SECTION_LABELS[upload_section]}."
                if upload_group:
                    message_text += f" Group: {upload_group}."
                if skipped_count:
                    message_text += f" Skipped {skipped_count} unsupported file(s)."
                message = message_block(message_text, "success")
            elif trigger == "move-files-btn":
                moved = move_files(selected_ids, move_target_section, move_target_group or "", dataset_slug)
                move_target = f"{SECTION_LABELS[move_target_section]}"
                if move_target_group:
                    move_target += f" / {move_target_group}"
                message = message_block(f"Moved {len(moved)} file(s) to {dataset_label} / {move_target}.", "success")
            elif trigger == "delete-files-btn":
                deleted = delete_files(selected_ids)
                message = message_block(f"Deleted {len(deleted)} file(s).", "danger")
            elif trigger == "refresh-files-btn":
                message = message_block(f"Refreshed {dataset_label}.", "info")
        except Exception as exc:
            message = message_block(str(exc), "danger")

        rows = list_audio_files(dataset_slug)
        return (
            rows,
            [],
            message,
            build_file_summary_cards(rows),
            list_group_summary_rows(dataset_slug),
            make_columns(["section", "group", "files", "good_files", "bad_files"]),
            build_dataset_banner(dataset_slug),
        )

    @app.callback(
        Output("saved-run-dropdown", "options"),
        Input("current-run-store", "data"),
    )
    def refresh_run_options(_current_run):
        return list_run_artifacts()

    @app.callback(
        Output("processing-message", "children"),
        Output("current-run-store", "data"),
        Input("run-pipeline-btn", "n_clicks"),
        State("run-mode", "value"),
        State("saved-run-dropdown", "value"),
        State("preprocessing-flags", "value"),
        State("lowcut-hz", "value"),
        State("highcut-hz", "value"),
        State("filter-order", "value"),
        State("pre-sec", "value"),
        State("post-sec", "value"),
        State("min-gap-sec", "value"),
        State("hop-length", "value"),
        State("delta-list", "value"),
        State("active-dataset-dropdown", "value"),
        State("bad-hits", "value"),
        State("model-checklist", "value"),
        State("feature-set-checklist", "value"),
        State("save-processed-hits", "value"),
        prevent_initial_call=True,
    )
    def run_pipeline(
        _n_clicks,
        run_mode,
        saved_run_path,
        preprocessing_flags,
        lowcut_hz,
        highcut_hz,
        filter_order,
        pre_sec,
        post_sec,
        min_gap_sec,
        hop_length,
        delta_list,
        active_dataset_slug,
        bad_hits,
        selected_models,
        selected_feature_sets,
        save_processed_hits_flag,
    ):
        dataset_slug = active_dataset_slug or default_active_dataset
        dataset_label = get_dataset_record(dataset_slug)["label"]
        try:
            if run_mode == "use_saved":
                if not saved_run_path:
                    raise ValueError("Choose a saved run before selecting 'Use an existing saved run'.")
                saved_bundle = load_run_artifact(saved_run_path)
                bundle = evaluate_saved_experiment(saved_bundle, dataset_slug)
            else:
                config = build_experiment_config(
                    preprocessing_flags or [],
                    lowcut_hz,
                    highcut_hz,
                    filter_order,
                    pre_sec,
                    post_sec,
                    min_gap_sec,
                    hop_length,
                    delta_list,
                    bad_hits,
                    selected_models or [],
                    selected_feature_sets or [],
                    save_processed_hits_flag or [],
                )
                bundle = train_experiment(config, dataset_slug)

            artifact_path = save_run_artifact(bundle)
            top_row = bundle["results_table"][0] if bundle["results_table"] else None
            summary = f"Saved run {bundle['run_id']} for {dataset_label} to {artifact_path.name}."
            if top_row:
                summary += (
                    f" Best validation model: {top_row['feature_set']} | {top_row['model']} "
                    f"({top_row['validation_accuracy']:.4f})."
                )
            return message_block(summary, "success"), {"artifact_path": str(artifact_path), "run_id": bundle["run_id"]}
        except Exception as exc:
            return message_block(str(exc), "danger"), no_update

    @app.callback(
        Output("preview-file-dropdown", "options"),
        Output("preview-file-dropdown", "value"),
        Input("file-table", "data"),
        State("preview-file-dropdown", "value"),
    )
    def sync_preview_dropdown(rows, current_value):
        rows = rows or []
        options = [{"label": f"{row['section']} / {row['group']} / {row['name']}", "value": row["file_id"]} for row in rows]
        values = {option["value"] for option in options}
        if current_value in values:
            return options, current_value
        return options, (options[0]["value"] if options else None)

    @app.callback(
        Output("preview-audio", "src"),
        Output("preview-metadata", "children"),
        Output("raw-waveform-graph", "figure"),
        Output("processed-waveform-graph", "figure"),
        Output("onset-graph", "figure"),
        Output("spectrogram-graph", "figure"),
        Output("clip-lengths-graph", "figure"),
        Input("preview-file-dropdown", "value"),
        State("preprocessing-flags", "value"),
        State("lowcut-hz", "value"),
        State("highcut-hz", "value"),
        State("filter-order", "value"),
        State("pre-sec", "value"),
        State("post-sec", "value"),
        State("min-gap-sec", "value"),
        State("hop-length", "value"),
        State("delta-list", "value"),
        State("bad-hits", "value"),
    )
    def update_signal_preview(
        file_id,
        preprocessing_flags,
        lowcut_hz,
        highcut_hz,
        filter_order,
        pre_sec,
        post_sec,
        min_gap_sec,
        hop_length,
        delta_list,
        bad_hits,
    ):
        empty_wave = make_waveform_figure([], [], "No file selected", "#9ca3af")
        empty_onset = make_onset_figure([], [], [])
        empty_spec = make_spectrogram_figure([[0]], [0], [0])
        empty_clips = make_clip_lengths_figure([])

        if not file_id:
            return "", "Select a file to inspect.", empty_wave, empty_wave, empty_onset, empty_spec, empty_clips

        dataset_slug, section, relative_path = file_id.split("|", 2)
        file_path = resolve_audio_path(dataset_slug, section, relative_path)
        config = build_experiment_config(
            preprocessing_flags or [],
            lowcut_hz,
            highcut_hz,
            filter_order,
            pre_sec,
            post_sec,
            min_gap_sec,
            hop_length,
            delta_list,
            bad_hits,
            ["KNN"],
            ["PSD"],
            ["save"],
        )

        try:
            preview = build_signal_preview(file_path, config.preprocessing, config.split, config.bad_hits)
            metadata = (
                f"{get_dataset_record(dataset_slug)['label']} / {section.title()} / {relative_path} | "
                f"Label: {preview['label']} | Expected hits: {preview['expected_hits']} | "
                f"Found hits: {preview['found_hits']} | Delta used: {preview['delta_used']} | Duration: {preview['duration_sec']}s"
            )
            return (
                make_media_url(file_id),
                metadata,
                make_waveform_figure(preview["raw_time"], preview["raw_signal"], "Raw Waveform", "#c2410c"),
                make_waveform_figure(preview["processed_time"], preview["processed_signal"], "Processed Waveform", "#0f766e"),
                make_onset_figure(preview["onset_time_axis"], preview["onset_envelope"], preview["onset_times"]),
                make_spectrogram_figure(preview["spectrogram_db"], preview["spectrogram_times"], preview["spectrogram_freqs"]),
                make_clip_lengths_figure(preview["clip_lengths"]),
            )
        except Exception as exc:
            error_figure = make_waveform_figure([], [], "Preview unavailable", "#9ca3af")
            return make_media_url(file_id), str(exc), error_figure, error_figure, empty_onset, empty_spec, empty_clips

    @app.callback(
        Output("run-banner-container", "children"),
        Output("accuracy-graph", "figure"),
        Output("results-table", "data"),
        Output("results-table", "columns"),
        Output("robustness-table", "data"),
        Output("robustness-table", "columns"),
        Output("tuning-table", "data"),
        Output("tuning-table", "columns"),
        Output("confusion-grid", "children"),
        Output("diagnostics-table", "data"),
        Output("diagnostics-table", "columns"),
        Input("current-run-store", "data"),
    )
    def populate_results(current_run):
        if not current_run or not current_run.get("artifact_path"):
            empty_results = []
            empty_columns = make_columns([])
            return (
                message_block("Run the pipeline to populate analysis outputs.", "info"),
                make_accuracy_figure([]),
                empty_results,
                empty_columns,
                empty_results,
                empty_columns,
                empty_results,
                empty_columns,
                [],
                empty_results,
                empty_columns,
            )

        bundle = load_run_artifact(current_run["artifact_path"])
        results_rows = bundle.get("results_table", [])
        robustness_rows = bundle.get("robustness_table", [])
        tuning_rows = bundle.get("tuning_table", [])
        diagnostics_rows = flatten_diagnostics(bundle)

        return (
            build_run_banner(bundle),
            make_accuracy_figure(results_rows),
            results_rows,
            make_columns(list(results_rows[0].keys()) if results_rows else []),
            robustness_rows,
            make_columns(list(robustness_rows[0].keys()) if robustness_rows else []),
            tuning_rows,
            make_columns(list(tuning_rows[0].keys()) if tuning_rows else []),
            build_confusion_grid(bundle),
            diagnostics_rows,
            make_columns(list(diagnostics_rows[0].keys()) if diagnostics_rows else []),
        )
