from __future__ import annotations

from urllib.parse import quote

from dash import dcc, html

from .config import DEFAULT_DATASET_LABEL, SECTION_LABELS, SECTION_NAMES
from .plots import make_confusion_matrix_figure
from .storage import get_dataset_record


def message_block(message: str, tone: str = "info") -> html.Div:
    return html.Div(message, className=f"message message-{tone}")


def make_columns(column_names: list[str]) -> list[dict[str, str]]:
    return [{"name": name.replace("_", " ").title(), "id": name} for name in column_names]


def build_file_summary_cards(rows: list[dict]) -> list[html.Div]:
    summaries = {section: {"files": 0, "good": 0, "bad": 0, "groups": set()} for section in SECTION_NAMES}

    for row in rows:
        section = row["section_key"]
        summaries[section]["files"] += 1
        summaries[section]["groups"].add(row["group_key"] or "Root")
        if row["label_hint"] == "good":
            summaries[section]["good"] += 1
        elif row["label_hint"] == "bad":
            summaries[section]["bad"] += 1

    cards: list[html.Div] = []
    for section in SECTION_NAMES:
        data = summaries[section]
        cards.append(
            html.Div(
                className="summary-card",
                children=[
                    html.Div(SECTION_LABELS[section], className="summary-title"),
                    html.Div(str(data["files"]), className="summary-value"),
                    html.Div(
                        f"Groups: {len(data['groups'])} | Good: {data['good']} | Bad: {data['bad']}",
                        className="summary-meta",
                    ),
                ],
            )
        )
    return cards


def build_dataset_banner(dataset_slug: str) -> html.Div:
    dataset = get_dataset_record(dataset_slug)
    description = dataset.get("description") or "Selectable dataset bundle for recording, file management, processing, and results."
    return html.Div(
        className="dataset-banner",
        children=[
            html.Div(dataset["label"], className="run-banner-title"),
            html.Div(
                f"Files: {dataset['file_count']} | Groups: {dataset['group_count']} | "
                f"Training: {dataset['training_files']} | Testing: {dataset['testing_files']} | Validation: {dataset['validation_files']}",
                className="run-banner-mode",
            ),
            html.Div(description, className="run-banner-meta"),
        ],
    )


def build_run_banner(bundle: dict) -> html.Div:
    results = bundle.get("results_table", [])
    top_row = results[0] if results else None
    top_text = "No model results yet."
    if top_row:
        top_text = (
            f"Best validation accuracy: {top_row['feature_set']} | {top_row['model']} "
            f"= {top_row['validation_accuracy']:.4f}"
        )

    source_text = ""
    if bundle.get("mode") == "loaded-evaluation" and bundle.get("source_run_id"):
        source_text = f" | Evaluated from saved run {bundle['source_run_id']}"

    dataset_text = bundle.get("dataset_label") or DEFAULT_DATASET_LABEL

    return html.Div(
        className="run-banner",
        children=[
            html.Div(f"Current run: {bundle.get('run_id', 'None')}", className="run-banner-title"),
            html.Div(
                f"{bundle.get('mode', 'idle').replace('-', ' ').title()} | Dataset: {dataset_text}{source_text}",
                className="run-banner-mode",
            ),
            html.Div(top_text, className="run-banner-meta"),
        ],
    )


def flatten_diagnostics(bundle: dict) -> list[dict]:
    rows: list[dict] = []
    for section in SECTION_NAMES:
        for row in bundle.get("section_diagnostics", {}).get(section, []):
            merged = {"section": SECTION_LABELS[section]}
            merged.update(row)
            rows.append(merged)
    return rows


def build_confusion_grid(bundle: dict) -> list[html.Div]:
    cards: list[html.Div] = []
    confusion_matrices = bundle.get("confusion_matrices", {})
    results_lookup = {
        (row["feature_set"], row["model"]): row
        for row in bundle.get("results_table", [])
    }

    for feature_name, models in confusion_matrices.items():
        for model_name, matrices in models.items():
            result_row = results_lookup.get((feature_name, model_name), {})
            cards.append(
                html.Div(
                    className="panel",
                    children=[
                        html.Div(
                            className="panel-heading",
                            children=[
                                html.H3(f"{feature_name} | {model_name}"),
                                html.P(
                                    f"Validation: {result_row.get('validation_accuracy', 0):.4f} | "
                                    f"Best Params: {result_row.get('best_params', '{}')}"
                                ),
                            ],
                        ),
                        html.Div(
                            className="graph-grid three-up",
                            children=[
                                dcc.Graph(
                                    figure=make_confusion_matrix_figure(matrices["train"], "Training Confusion Matrix"),
                                    config={"displayModeBar": False},
                                ),
                                dcc.Graph(
                                    figure=make_confusion_matrix_figure(matrices["testing"], "Testing Confusion Matrix"),
                                    config={"displayModeBar": False},
                                ),
                                dcc.Graph(
                                    figure=make_confusion_matrix_figure(matrices["validation"], "Validation Confusion Matrix"),
                                    config={"displayModeBar": False},
                                ),
                            ],
                        ),
                    ],
                )
            )
    return cards


def make_media_url(file_id: str | None) -> str:
    if not file_id:
        return ""
    dataset_slug, section, relative_path = file_id.split("|", 2)
    encoded_path = "/".join(quote(part) for part in relative_path.split("/"))
    return f"/media/{quote(dataset_slug)}/{quote(section)}/{encoded_path}"
