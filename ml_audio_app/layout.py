from __future__ import annotations

from dash import dash_table, dcc, html

from .config import DEFAULT_BAD_HITS, DEFAULT_RECORDING_PREFIX, SECTION_LABELS, SECTION_NAMES
from .storage import get_latest_run_artifact, list_audio_files, list_dataset_options, list_group_summary_rows, list_run_artifacts
from .ui_helpers import build_attached_run_status, build_dataset_banner, build_file_summary_cards, make_columns, message_block


def studio_tab(default_dataset_slug: str) -> html.Div:
    rows = list_audio_files(default_dataset_slug)
    return html.Div(
        className="tab-content",
        children=[
            html.Div(id="active-dataset-banner", children=build_dataset_banner(default_dataset_slug)),
            html.Div(
                className="panel studio-hero",
                children=[
                    html.Div(
                        className="hero-copy",
                        children=[
                            html.P("Studio", className="eyebrow"),
                            html.H2("Capture new recordings with the assignment naming convention."),
                            html.P(
                                "Browser recordings are saved as WAV for compatibility. Uploaded files can still be M4A or WAV."
                            ),
                        ],
                    ),
                    html.Div(
                        className="hero-side",
                        children=[
                            html.Div(
                                id="dataset-summary-cards",
                                className="summary-grid",
                                children=build_file_summary_cards(rows),
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="panel",
                children=[
                    html.Div(
                        className="form-grid",
                        children=[
                            html.Div(
                                children=[
                                    html.Label("Prefix"),
                                    dcc.Input(id="record-prefix", type="text", value=DEFAULT_RECORDING_PREFIX, className="control"),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Sample Number"),
                                    dcc.Input(id="record-sample-number", type="number", value=1, min=1, max=27, step=1, className="control"),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Condition"),
                                    dcc.Dropdown(
                                        id="record-condition",
                                        options=[
                                            {"label": "Healthy / Good (_g)", "value": "g"},
                                            {"label": "Unhealthy / Bad (_b)", "value": "b"},
                                        ],
                                        value="g",
                                        clearable=False,
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Target Section"),
                                    dcc.Dropdown(
                                        id="record-target-section",
                                        options=[{"label": SECTION_LABELS[key], "value": key} for key in SECTION_NAMES],
                                        value="training",
                                        clearable=False,
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Target Group / Folder"),
                                    dcc.Input(
                                        id="record-group",
                                        type="text",
                                        value="",
                                        placeholder="Optional, e.g. Set6 or Day2/NewBatch",
                                        className="control",
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(id="recording-filename-preview", className="filename-preview"),
                    html.Div(
                        className="button-row",
                        children=[
                            html.Button("Start Recording", id="record-start-btn", n_clicks=0, className="button button-primary"),
                            html.Button("Stop Recording", id="record-stop-btn", n_clicks=0, className="button button-secondary"),
                            html.Button("Save Recording", id="save-recording-btn", n_clicks=0, className="button button-accent"),
                        ],
                    ),
                    html.Div(id="recording-status", className="muted-text", children="Microphone is idle."),
                    html.Audio(id="recording-preview-audio", controls=True, className="audio-player"),
                    dcc.Textarea(id="recording-data", style={"display": "none"}),
                    html.Div(id="studio-message"),
                ],
            ),
        ],
    )


def file_management_tab(default_dataset_slug: str) -> html.Div:
    rows = list_audio_files(default_dataset_slug)
    group_rows = list_group_summary_rows(default_dataset_slug)
    latest_run = get_latest_run_artifact(default_dataset_slug)
    current_run = {"artifact_path": latest_run["value"], "load_mode": "attached_saved"} if latest_run else None
    return html.Div(
        className="tab-content",
        children=[
            html.Div(
                className="panel",
                children=[
                    html.Div(
                        className="panel-heading",
                        children=[
                            html.H2("File Management"),
                            html.P(
                                "Work with dataset bundles that contain training, testing, and validation sections plus nested folders like Set1-Set5 or independent capture days."
                            ),
                        ],
                    ),
                    html.Div(id="attached-run-status-file", children=build_attached_run_status(default_dataset_slug, current_run)),
                    html.Div(className="dataset-selector-note", children="The active dataset picker in the header controls what you see here."),
                    html.Div(
                        className="upload-row",
                        children=[
                            html.Div(
                                className="upload-card",
                                children=[
                                    html.Label("Upload Destination Section"),
                                    dcc.Dropdown(
                                        id="upload-section",
                                        options=[{"label": SECTION_LABELS[key], "value": key} for key in SECTION_NAMES],
                                        value="training",
                                        clearable=False,
                                    ),
                                    html.Label("Group / Subfolder"),
                                    dcc.Input(
                                        id="upload-group",
                                        type="text",
                                        value="",
                                        placeholder="Examples: Set6, Day2, BatchA/Retest",
                                        className="control",
                                    ),
                                    dcc.Upload(
                                        id="file-upload",
                                        multiple=True,
                                        className="upload-dropzone",
                                        children=html.Div(
                                            [
                                                html.Span("Drag and drop audio files here"),
                                                html.Br(),
                                                html.Small("or click to browse"),
                                            ]
                                        ),
                                    ),
                                ],
                            ),
                            html.Div(
                                className="action-card",
                                children=[
                                    html.Label("Selected File Actions"),
                                    dcc.Dropdown(
                                        id="move-target-section",
                                        options=[{"label": SECTION_LABELS[key], "value": key} for key in SECTION_NAMES],
                                        value="testing",
                                        clearable=False,
                                    ),
                                    html.Label("Move Into Group / Folder"),
                                    dcc.Input(
                                        id="move-target-group",
                                        type="text",
                                        value="",
                                        placeholder="Optional target group",
                                        className="control",
                                    ),
                                    html.Div(
                                        className="button-row wrap",
                                        children=[
                                            html.Button("Refresh", id="refresh-files-btn", n_clicks=0, className="button button-secondary"),
                                            html.Button("Move Selected", id="move-files-btn", n_clicks=0, className="button"),
                                            html.Button("Delete Selected", id="delete-files-btn", n_clicks=0, className="button button-danger"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(id="file-management-message"),
                    dash_table.DataTable(
                        id="group-summary-table",
                        columns=make_columns(["section", "group", "files", "good_files", "bad_files"]),
                        data=group_rows,
                        sort_action="native",
                        page_size=8,
                        style_table={"overflowX": "auto", "marginBottom": "18px"},
                        style_cell={"textAlign": "left", "padding": "10px"},
                        style_header={"fontWeight": "bold"},
                    ),
                    dash_table.DataTable(
                        id="file-table",
                        columns=make_columns(["section", "group", "name", "label_hint", "size_kb", "modified"]),
                        data=rows,
                        row_selectable="multi",
                        selected_rows=[],
                        sort_action="native",
                        filter_action="native",
                        page_size=14,
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "10px"},
                        style_header={"fontWeight": "bold"},
                    ),
                ],
            ),
        ],
    )


def processing_tab(default_dataset_slug: str) -> html.Div:
    latest_run = get_latest_run_artifact(default_dataset_slug)
    default_run_mode = "use_saved" if latest_run else "train_new"
    default_saved_run = latest_run["value"] if latest_run else None
    attached_run_note = (
        "Latest saved run is attached automatically for this dataset. Choose 'Train new models' only when you want to rebuild features and refit classifiers."
        if latest_run
        else "No saved run is attached yet for this dataset. Run training once to create a reusable model bundle."
    )

    return html.Div(
        className="tab-content",
        children=[
            html.Div(
                className="panel",
                children=[
                    html.Div(
                        className="panel-heading",
                        children=[
                            html.H2("Processing"),
                            html.P("Control preprocessing, hit splitting, feature extraction, and model execution."),
                        ],
                    ),
                    html.Div(
                        className="form-grid two-column",
                        children=[
                            html.Div(
                                className="stack",
                                children=[
                                    html.Label("Run Mode"),
                                    dcc.RadioItems(
                                        id="run-mode",
                                        options=[
                                            {"label": "Train new models", "value": "train_new"},
                                            {"label": "Use an existing saved run", "value": "use_saved"},
                                        ],
                                        value=default_run_mode,
                                        className="inline-options",
                                    ),
                                    html.Label("Saved Run"),
                                    dcc.Dropdown(
                                        id="saved-run-dropdown",
                                        options=list_run_artifacts(default_dataset_slug),
                                        value=default_saved_run,
                                        placeholder="Select a saved run",
                                    ),
                                    html.Div(attached_run_note, className="muted-text"),
                                    html.Label("Preprocessing"),
                                    dcc.Checklist(
                                        id="preprocessing-flags",
                                        options=[
                                            {"label": "Enable preprocessing", "value": "enabled"},
                                            {"label": "Remove DC offset", "value": "remove_dc"},
                                            {"label": "Normalize peak", "value": "normalize_peak"},
                                            {"label": "Apply bandpass", "value": "apply_bandpass"},
                                        ],
                                        value=["enabled", "remove_dc", "normalize_peak", "apply_bandpass"],
                                        className="stacked-options",
                                    ),
                                    html.Label("Feature Sets"),
                                    dcc.Checklist(
                                        id="feature-set-checklist",
                                        options=[{"label": "PSD", "value": "PSD"}, {"label": "MFCC", "value": "MFCC"}],
                                        value=["PSD", "MFCC"],
                                        className="inline-options",
                                    ),
                                    html.Label("Models"),
                                    dcc.Checklist(
                                        id="model-checklist",
                                        options=[
                                            {"label": "KNN", "value": "KNN"},
                                            {"label": "Decision Tree", "value": "Decision Tree"},
                                            {"label": "Logistic Regression", "value": "Logistic Regression"},
                                            {"label": "SVM", "value": "SVM"},
                                        ],
                                        value=["KNN", "Decision Tree", "Logistic Regression", "SVM"],
                                        className="stacked-options",
                                    ),
                                    html.Label("Artifacts"),
                                    dcc.Checklist(
                                        id="save-processed-hits",
                                        options=[{"label": "Save split single-hit clips under artifacts/processed_hits", "value": "save"}],
                                        value=["save"],
                                        className="stacked-options",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="stack",
                                children=[
                                    html.Div(
                                        className="numeric-grid",
                                        children=[
                                            html.Div(children=[html.Label("Low Cut (Hz)"), dcc.Input(id="lowcut-hz", type="number", value=80.0, step=1, className="control")]),
                                            html.Div(children=[html.Label("High Cut (Hz)"), dcc.Input(id="highcut-hz", type="number", value=8000.0, step=1, className="control")]),
                                            html.Div(children=[html.Label("Filter Order"), dcc.Input(id="filter-order", type="number", value=4, step=1, min=1, className="control")]),
                                            html.Div(children=[html.Label("Bad Hits per Recording"), dcc.Input(id="bad-hits", type="number", value=DEFAULT_BAD_HITS, step=1, min=1, className="control")]),
                                            html.Div(children=[html.Label("Pre Window (s)"), dcc.Input(id="pre-sec", type="number", value=0.02, step=0.01, className="control")]),
                                            html.Div(children=[html.Label("Post Window (s)"), dcc.Input(id="post-sec", type="number", value=0.20, step=0.01, className="control")]),
                                            html.Div(children=[html.Label("Min Gap (s)"), dcc.Input(id="min-gap-sec", type="number", value=0.04, step=0.01, className="control")]),
                                            html.Div(children=[html.Label("Hop Length"), dcc.Input(id="hop-length", type="number", value=256, step=1, min=32, className="control")]),
                                        ],
                                    ),
                                    html.Label("Adaptive Delta List"),
                                    dcc.Textarea(
                                        id="delta-list",
                                        value="0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04",
                                        className="control control-textarea",
                                    ),
                                    html.Div(className="button-row", children=[html.Button("Run Processing Pipeline", id="run-pipeline-btn", n_clicks=0, className="button button-primary")]),
                                    html.Div(id="processing-message"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def results_tab(default_dataset_slug: str) -> html.Div:
    latest_run = get_latest_run_artifact(default_dataset_slug)
    current_run = {"artifact_path": latest_run["value"], "load_mode": "attached_saved"} if latest_run else None
    return html.Div(
        className="tab-content",
        children=[
            html.Div(id="attached-run-status-results", children=build_attached_run_status(default_dataset_slug, current_run)),
            html.Div(id="run-banner-container", children=message_block("Run the pipeline to populate analysis outputs.", "info")),
            html.Div(
                className="panel",
                children=[
                    html.Div(
                        className="panel-heading",
                        children=[
                            html.H2("Signal Preview"),
                            html.P("Inspect a single recording with waveform, preprocessing, onset detection, and clip durations."),
                        ],
                    ),
                    dcc.Dropdown(id="preview-file-dropdown", placeholder="Select a file from Training, Testing, or Validation"),
                    html.Audio(id="preview-audio", controls=True, className="audio-player"),
                    html.Div(id="preview-metadata", className="muted-text"),
                    html.Div(
                        className="graph-grid",
                        children=[
                            dcc.Graph(id="raw-waveform-graph"),
                            dcc.Graph(id="processed-waveform-graph"),
                            dcc.Graph(id="onset-graph"),
                            dcc.Graph(id="spectrogram-graph"),
                            dcc.Graph(id="clip-lengths-graph"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="panel",
                children=[
                    html.Div(
                        className="panel-heading",
                        children=[
                            html.H2("Model Results"),
                            html.P("Compare training, testing, and validation performance across the selected classifiers."),
                        ],
                    ),
                    dcc.Graph(id="accuracy-graph"),
                    dash_table.DataTable(
                        id="results-table",
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "10px"},
                        style_header={"fontWeight": "bold"},
                        page_size=10,
                    ),
                    html.H3("Robustness Ranking"),
                    dash_table.DataTable(
                        id="robustness-table",
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "10px"},
                        style_header={"fontWeight": "bold"},
                        page_size=8,
                    ),
                    html.H3("Top Hyperparameter Trials"),
                    dash_table.DataTable(
                        id="tuning-table",
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "10px"},
                        style_header={"fontWeight": "bold"},
                        page_size=8,
                    ),
                ],
            ),
            html.Div(id="confusion-grid", className="stack"),
            html.Div(
                className="panel",
                children=[
                    html.Div(
                        className="panel-heading",
                        children=[
                            html.H2("Section Diagnostics"),
                            html.P("See how many hits were found versus expected for each source file."),
                        ],
                    ),
                    dash_table.DataTable(
                        id="diagnostics-table",
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "10px"},
                        style_header={"fontWeight": "bold"},
                        page_size=10,
                    ),
                ],
            ),
        ],
    )


def create_layout(default_dataset_slug: str) -> html.Div:
    latest_run = get_latest_run_artifact(default_dataset_slug)
    initial_run_store = {"artifact_path": latest_run["value"], "load_mode": "attached_saved"} if latest_run else None

    return html.Div(
        className="app-shell",
        children=[
            dcc.Store(id="current-run-store", data=initial_run_store),
            html.Header(
                className="app-header",
                children=[
                    html.Div(
                        className="header-grid",
                        children=[
                            html.Div(
                                className="header-copy",
                                children=[
                                    html.P("ML Audio Processing App", className="eyebrow"),
                                    html.H1("Studio-to-Validation Workflow Suite for Experimental impact audio ML classification"),
                                    html.P("Built from the notebook pipeline using Dash, Plotly, librosa, and scikit-learn."),
                                ],
                            ),
                            html.Div(
                                className="dataset-picker",
                                children=[
                                    html.Label("Active Dataset Bundle"),
                                    dcc.Dropdown(
                                        id="active-dataset-dropdown",
                                        options=list_dataset_options(),
                                        value=default_dataset_slug,
                                        clearable=False,
                                    ),
                                    html.Div(
                                        "Switch datasets here to browse, process, and demo different collections.",
                                        className="muted-text",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Tabs(
                id="main-tabs",
                value="studio",
                className="main-tabs",
                children=[
                    dcc.Tab(label="Studio", value="studio", children=studio_tab(default_dataset_slug)),
                    dcc.Tab(label="File Management", value="file_management", children=file_management_tab(default_dataset_slug)),
                    dcc.Tab(label="Processing", value="processing", children=processing_tab(default_dataset_slug)),
                    dcc.Tab(label="Results / Analysis", value="results", children=results_tab(default_dataset_slug)),
                ],
            ),
        ],
    )
