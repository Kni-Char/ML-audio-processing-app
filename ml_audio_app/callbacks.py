from __future__ import annotations

import json
from pathlib import Path

from dash import ALL, ClientsideFunction, Input, Output, State, callback_context, no_update

from .config import DEFAULT_RECORDING_PREFIX, DEFAULT_TRAIN_RATIO, SOURCE_SECTION_LABELS
from .pipeline import build_signal_preview, train_experiment
from .plots import (
    make_accuracy_figure,
    make_clip_lengths_figure,
    make_onset_figure,
    make_spectrogram_figure,
    make_waveform_figure,
)
from .storage import (
    build_recording_name,
    delete_folders,
    create_subfolder,
    delete_files,
    get_dataset_record,
    get_latest_run_artifact,
    list_audio_files,
    list_folder_records,
    list_run_artifacts,
    load_run_artifact,
    resolve_audio_path,
    save_recording,
    save_run_artifact,
    save_uploaded_files,
)
from .ui_helpers import (
    build_confusion_grid,
    build_attached_run_status,
    build_browser_rows,
    build_dataset_banner,
    build_file_summary_cards,
    build_file_tree,
    build_run_banner,
    list_available_folder_keys,
    parse_folder_key,
    flatten_diagnostics,
    make_columns,
    make_media_url,
    message_block,
)
from .workflow import build_experiment_config


def register_callbacks(app, default_active_dataset: str) -> None:
    app.clientside_callback(
        ClientsideFunction(namespace="clientside", function_name="setProcessingStartedAt"),
        Output("processing-started-at", "data"),
        Input("run-pipeline-btn", "n_clicks"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        ClientsideFunction(namespace="clientside", function_name="updateProcessingTimer"),
        Output("processing-timer", "children"),
        Output("processing-progress-note", "children"),
        Input("processing-started-at", "data"),
        Input("processing-timer-interval", "n_intervals"),
    )

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
        section_label = SOURCE_SECTION_LABELS.get(section or "pooled", "Train/Test Pool")
        group_label = f" / {group}" if group else ""
        return f"Next saved recording: {dataset_label} / {section_label}{group_label} / {name}"

    @app.callback(
        Output("studio-message", "children"),
        Output("dataset-summary-cards", "children", allow_duplicate=True),
        Output("active-dataset-banner", "children", allow_duplicate=True),
        Output("file-manager-refresh-token", "data", allow_duplicate=True),
        Input("save-recording-btn", "n_clicks"),
        State("recording-data", "value"),
        State("active-dataset-dropdown", "value"),
        State("record-target-section", "value"),
        State("record-group", "value"),
        State("record-prefix", "value"),
        State("record-sample-number", "value"),
        State("record-condition", "value"),
        State("file-manager-refresh-token", "data"),
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
        refresh_token,
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
                message_block(f"Saved recording to {get_dataset_record(dataset_slug)['label']} / {SOURCE_SECTION_LABELS[section]} as {destination.name}.", "success"),
                build_file_summary_cards(rows),
                build_dataset_banner(dataset_slug),
                int(refresh_token or 0) + 1,
            )
        except Exception as exc:
            rows = list_audio_files(dataset_slug or default_active_dataset)
            active_slug = dataset_slug or default_active_dataset
            return (
                message_block(str(exc), "danger"),
                build_file_summary_cards(rows),
                build_dataset_banner(active_slug),
                int(refresh_token or 0),
            )

    @app.callback(
        Output("file-management-message", "children"),
        Output("dataset-summary-cards", "children", allow_duplicate=True),
        Output("active-dataset-banner", "children", allow_duplicate=True),
        Output("file-manager-refresh-token", "data", allow_duplicate=True),
        Output("create-folder-name", "value"),
        Output("file-upload-relative-paths", "value"),
        Input("active-dataset-dropdown", "value"),
        Input("file-upload", "contents"),
        Input("refresh-files-btn", "n_clicks"),
        Input("create-folder-btn", "n_clicks"),
        Input("delete-files-btn", "n_clicks"),
        State("file-upload", "filename"),
        State("file-upload-relative-paths", "value"),
        State("selected-folder-store", "data"),
        State("create-folder-name", "value"),
        State("file-table", "data"),
        State("file-table", "derived_virtual_data"),
        State("file-table", "selected_rows"),
        State("file-manager-refresh-token", "data"),
        prevent_initial_call=True,
    )
    def handle_file_actions(
        active_dataset_slug,
        upload_contents,
        _refresh_clicks,
        _create_folder_clicks,
        _delete_clicks,
        upload_filenames,
        upload_relative_paths_raw,
        selected_folder_key,
        create_folder_name,
        table_data,
        visible_rows,
        selected_rows,
        refresh_token,
    ):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
        dataset_slug = active_dataset_slug or default_active_dataset
        dataset_label = get_dataset_record(dataset_slug)["label"]
        message = message_block(f"{dataset_label} is ready for uploads and section moves.", "info")
        current_rows = visible_rows or table_data or list_audio_files(dataset_slug)
        selected_rows = selected_rows or []
        selected_file_ids = []
        selected_folder_targets = []
        for index in selected_rows:
            if index >= len(current_rows):
                continue
            row = current_rows[index]
            if row.get("row_type") == "file" and row.get("file_id"):
                selected_file_ids.append(row["file_id"])
            elif row.get("row_type") == "folder" and row.get("nav_key"):
                folder_section, folder_group = parse_folder_key(row["nav_key"])
                if folder_section and folder_group:
                    selected_folder_targets.append((folder_section, folder_group))

        try:
            if trigger == "active-dataset-dropdown":
                message = message_block(f"Showing files from {dataset_label}.", "info")
            if trigger == "file-upload" and upload_contents:
                upload_section, upload_group = parse_folder_key(selected_folder_key)
                if not upload_section:
                    raise ValueError("Select a destination folder in the tree before uploading.")
                try:
                    upload_relative_paths = json.loads(upload_relative_paths_raw) if upload_relative_paths_raw else []
                except json.JSONDecodeError:
                    upload_relative_paths = []
                result = save_uploaded_files(upload_contents, upload_filenames, dataset_slug, upload_section, upload_group or "", upload_relative_paths)
                saved_count = len(result["saved"])
                skipped_count = len(result["skipped"])
                message_text = f"Uploaded {saved_count} file(s) to {dataset_label} / {SOURCE_SECTION_LABELS[upload_section]}."
                if upload_group:
                    message_text += f" Folder: {upload_group}."
                elif upload_relative_paths and any("/" in path or "\\" in path for path in upload_relative_paths):
                    message_text += " Folder structure was preserved."
                if skipped_count:
                    message_text += f" Skipped {skipped_count} unsupported file(s)."
                message = message_block(message_text, "success")
            elif trigger == "create-folder-btn":
                target_section, target_group = parse_folder_key(selected_folder_key)
                if not target_section:
                    raise ValueError("Select Train/Test Pool, Validation, or a sub-folder before creating a new folder.")
                created = create_subfolder(dataset_slug, target_section, target_group or "", create_folder_name or "")
                parent_label = SOURCE_SECTION_LABELS[target_section]
                if target_group:
                    parent_label += f" / {target_group}"
                message = message_block(f"Created folder '{created.name}' inside {dataset_label} / {parent_label}.", "success")
            elif trigger == "delete-files-btn":
                if not selected_file_ids and not selected_folder_targets:
                    current_section, current_group = parse_folder_key(selected_folder_key)
                    if current_section and current_group:
                        selected_folder_targets.append((current_section, current_group))

                deleted_files = delete_files(selected_file_ids)
                deleted_folders = delete_folders(dataset_slug, selected_folder_targets)

                deleted_parts = []
                if deleted_files:
                    deleted_parts.append(f"{len(deleted_files)} file(s)")
                if deleted_folders:
                    deleted_parts.append(f"{len(deleted_folders)} folder(s)")
                if not deleted_parts:
                    raise ValueError("Select one or more files or a sub-folder before deleting. Root sections cannot be deleted.")
                message = message_block(f"Deleted {' and '.join(deleted_parts)} from {dataset_label}.", "success")
            elif trigger == "refresh-files-btn":
                message = message_block(f"Refreshed {dataset_label}.", "info")
        except Exception as exc:
            message = message_block(str(exc), "danger")

        rows = list_audio_files(dataset_slug)
        return (
            message,
            build_file_summary_cards(rows),
            build_dataset_banner(dataset_slug),
            int(refresh_token or 0) + 1,
            "",
            "",
        )

    @app.callback(
        Output("selected-folder-store", "data"),
        Input("active-dataset-dropdown", "value"),
        Input("file-manager-refresh-token", "data"),
        Input({"type": "file-tree-node", "key": ALL}, "n_clicks"),
        State("selected-folder-store", "data"),
        prevent_initial_call=False,
    )
    def sync_selected_folder(active_dataset_slug, _refresh_token, _folder_clicks, current_folder_key):
        dataset_slug = active_dataset_slug or default_active_dataset
        folder_records = list_folder_records(dataset_slug)
        available_keys = list_available_folder_keys(folder_records)
        if not available_keys:
            return None

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        if trigger.startswith("{"):
            try:
                clicked = json.loads(trigger)
                clicked_key = clicked.get("key")
                if clicked_key in available_keys:
                    return clicked_key
            except json.JSONDecodeError:
                pass

        if current_folder_key in available_keys:
            return current_folder_key
        return available_keys[0]

    @app.callback(
        Output("file-tree-container", "children"),
        Output("upload-target-note", "children"),
        Output("upload-directory-mode", "children"),
        Output("file-table", "data"),
        Output("file-table", "selected_rows"),
        Input("active-dataset-dropdown", "value"),
        Input("selected-folder-store", "data"),
        Input("file-manager-refresh-token", "data"),
    )
    def render_file_manager(active_dataset_slug, selected_folder_key, _refresh_token):
        dataset_slug = active_dataset_slug or default_active_dataset
        rows = list_audio_files(dataset_slug)
        folder_records = list_folder_records(dataset_slug)
        available_keys = list_available_folder_keys(folder_records)
        effective_folder_key = selected_folder_key if selected_folder_key in available_keys else (available_keys[0] if available_keys else None)
        upload_section, upload_group = parse_folder_key(effective_folder_key)
        if upload_section:
            destination = SOURCE_SECTION_LABELS[upload_section]
            if upload_group:
                destination += f" / {upload_group}"
            if upload_group:
                upload_note = f"Uploads will go into: {destination}"
                upload_mode = "subfolder"
            else:
                upload_note = (
                    f"Uploads will go into: {destination}. Root selected, so you can choose a single folder like Set1 "
                    f"or a parent folder that contains multiple bundles such as Set1, Set2, Set3."
                )
                upload_mode = "root"
        else:
            upload_note = "Select a folder in the tree to upload audio."
            upload_mode = "subfolder"
        browser_rows = build_browser_rows(rows, folder_records, effective_folder_key)
        return (
            build_file_tree(folder_records, effective_folder_key),
            upload_note,
            upload_mode,
            browser_rows,
            [],
        )

    @app.callback(
        Output("file-table", "selected_rows", allow_duplicate=True),
        Input("toggle-select-all-btn", "n_clicks"),
        State("file-table", "data"),
        State("file-table", "selected_rows"),
        prevent_initial_call=True,
    )
    def toggle_select_all_rows(_n_clicks, table_rows, selected_rows):
        if not table_rows:
            return []

        selectable_rows = [
            index
            for index, row in enumerate(table_rows)
            if row.get("row_type") in {"file", "folder"}
        ]
        if not selectable_rows:
            return []

        current_selection = sorted(index for index in (selected_rows or []) if index in selectable_rows)
        if current_selection == selectable_rows:
            return []
        return selectable_rows

    @app.callback(
        Output("selected-folder-store", "data", allow_duplicate=True),
        Input("file-table", "active_cell"),
        State("file-table", "data"),
        State("selected-folder-store", "data"),
        prevent_initial_call=True,
    )
    def navigate_from_file_table(active_cell, table_rows, current_folder_key):
        if not active_cell or not table_rows:
            return no_update
        row_index = active_cell.get("row")
        if row_index is None or row_index >= len(table_rows):
            return no_update

        row = table_rows[row_index]
        if row.get("row_type") in {"folder", "back"} and row.get("nav_key"):
            return row["nav_key"]
        return no_update

    @app.callback(
        Output("saved-run-dropdown", "options"),
        Output("saved-run-dropdown", "value"),
        Output("run-mode", "value"),
        Input("active-dataset-dropdown", "value"),
        Input("current-run-store", "data"),
        State("saved-run-dropdown", "value"),
    )
    def refresh_run_options(active_dataset_slug, current_run, current_saved_run):
        dataset_slug = active_dataset_slug or default_active_dataset
        options = list_run_artifacts(dataset_slug)
        option_values = {option["value"] for option in options}

        current_artifact = (current_run or {}).get("artifact_path")
        if current_artifact in option_values:
            selected_value = current_artifact
        elif current_saved_run in option_values:
            selected_value = current_saved_run
        else:
            latest = get_latest_run_artifact(dataset_slug)
            selected_value = latest["value"] if latest else None

        run_mode = "use_saved" if selected_value else "train_new"
        return options, selected_value, run_mode

    @app.callback(
        Output("current-run-store", "data"),
        Input("active-dataset-dropdown", "value"),
        prevent_initial_call=True,
    )
    def attach_latest_saved_run(active_dataset_slug):
        dataset_slug = active_dataset_slug or default_active_dataset
        latest = get_latest_run_artifact(dataset_slug)
        if latest:
            return {"artifact_path": latest["value"], "load_mode": "attached_saved"}
        return None

    @app.callback(
        Output("attached-run-status-file", "children"),
        Output("attached-run-status-results", "children"),
        Input("active-dataset-dropdown", "value"),
        Input("current-run-store", "data"),
    )
    def update_attached_run_status(active_dataset_slug, current_run):
        dataset_slug = active_dataset_slug or default_active_dataset
        status = build_attached_run_status(dataset_slug, current_run)
        return status, status

    @app.callback(
        Output("preprocessing-detail-container", "style"),
        Output("preprocessing-advanced-container", "style"),
        Input("preprocessing-enabled", "value"),
        Input("preprocessing-detail-flags", "value"),
    )
    def toggle_preprocessing_controls(preprocessing_enabled, preprocessing_detail_flags):
        enabled = "enabled" in (preprocessing_enabled or [])
        bandpass_enabled = "apply_bandpass" in (preprocessing_detail_flags or [])
        detail_style = {"display": "grid"} if enabled else {"display": "none"}
        advanced_style = {"display": "grid"} if enabled and bandpass_enabled else {"display": "none"}
        return detail_style, advanced_style

    @app.callback(
        Output("processing-message", "children"),
        Output("current-run-store", "data", allow_duplicate=True),
        Input("run-pipeline-btn", "n_clicks"),
        State("run-mode", "value"),
        State("saved-run-dropdown", "value"),
        State("preprocessing-enabled", "value"),
        State("preprocessing-detail-flags", "value"),
        State("lowcut-hz", "value"),
        State("highcut-hz", "value"),
        State("filter-order", "value"),
        State("train-ratio", "value"),
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
        running=[
            (Output("processing-progress-shell", "style"), {"display": "grid"}, {"display": "none"}),
            (Output("processing-timer-interval", "disabled"), False, True),
        ],
        prevent_initial_call=True,
    )
    def run_pipeline(
        _n_clicks,
        run_mode,
        saved_run_path,
        preprocessing_enabled,
        preprocessing_detail_flags,
        lowcut_hz,
        highcut_hz,
        filter_order,
        train_ratio,
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
        preprocessing_flags = list(preprocessing_enabled or []) + list(preprocessing_detail_flags or [])
        try:
            if run_mode == "use_saved":
                if not saved_run_path:
                    latest = get_latest_run_artifact(dataset_slug)
                    saved_run_path = latest["value"] if latest else None
                if not saved_run_path:
                    raise ValueError("No saved run is attached to this dataset yet. Switch to 'Train new models' once to create one.")
                bundle = load_run_artifact(saved_run_path)
                summary = f"Loaded attached saved run {bundle.get('run_id', Path(saved_run_path).stem)} for {dataset_label}."
                return message_block(summary, "success"), {
                    "artifact_path": str(saved_run_path),
                    "run_id": bundle.get("run_id"),
                    "load_mode": "attached_saved",
                }
            else:
                config = build_experiment_config(
                    preprocessing_flags or [],
                    lowcut_hz,
                    highcut_hz,
                    filter_order,
                    train_ratio,
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
                summary = f"Saved run {bundle['run_id']} for {dataset_label} to {artifact_path.name} using a pooled {config.train_ratio:.0%}/{1 - config.train_ratio:.0%} train/test split."
                if top_row:
                    summary += (
                        f" Best validation model: {top_row['feature_set']} | {top_row['model']} "
                        f"({top_row['validation_accuracy']:.4f})."
                    )
                return message_block(summary, "success"), {
                    "artifact_path": str(artifact_path),
                    "run_id": bundle["run_id"],
                    "load_mode": "trained_now",
                }
        except Exception as exc:
            return message_block(str(exc), "danger"), no_update

    @app.callback(
        Output("preview-file-dropdown", "options"),
        Output("preview-file-dropdown", "value"),
        Input("active-dataset-dropdown", "value"),
        Input("file-manager-refresh-token", "data"),
        State("preview-file-dropdown", "value"),
    )
    def sync_preview_dropdown(active_dataset_slug, _refresh_token, current_value):
        dataset_slug = active_dataset_slug or default_active_dataset
        rows = list_audio_files(dataset_slug)
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
        State("preprocessing-enabled", "value"),
        State("preprocessing-detail-flags", "value"),
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
        preprocessing_enabled,
        preprocessing_detail_flags,
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
        preprocessing_flags = list(preprocessing_enabled or []) + list(preprocessing_detail_flags or [])
        config = build_experiment_config(
            preprocessing_flags or [],
            lowcut_hz,
            highcut_hz,
            filter_order,
            DEFAULT_TRAIN_RATIO,
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
                f"{get_dataset_record(dataset_slug)['label']} / {SOURCE_SECTION_LABELS.get(section, section.title())} / {relative_path} | "
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
            build_run_banner(bundle, current_run),
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
