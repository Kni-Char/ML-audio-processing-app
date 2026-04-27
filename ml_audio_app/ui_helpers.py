from __future__ import annotations

from urllib.parse import quote

from dash import dcc, html

from .config import DEFAULT_DATASET_LABEL, SECTION_LABELS, SECTION_NAMES, SOURCE_SECTION_LABELS, SOURCE_SECTION_NAMES
from .plots import make_confusion_matrix_figure
from .storage import RUNS_SECTION_KEY, RUNS_SECTION_LABEL, get_dataset_record


def message_block(message: str, tone: str = "info", auto_dismiss: bool = False) -> html.Div:
    class_name = f"message message-{tone}"
    if auto_dismiss:
        class_name += " message-auto-dismiss"
    return html.Div(message, className=class_name)


def make_columns(column_names: list[str]) -> list[dict[str, str]]:
    labels = {
        "testing_accuracy": "Validation Accuracy",
        "validation_accuracy": "Testing Accuracy",
        "testing_to_validation_drop": "Validation To Testing Drop",
    }
    return [{"name": labels.get(name, name.replace("_", " ").title()), "id": name} for name in column_names]


def build_file_summary_cards(rows: list[dict]) -> list[html.Div]:
    summaries = {section: {"files": 0, "good": 0, "bad": 0, "groups": set()} for section in SOURCE_SECTION_NAMES}

    for row in rows:
        section = row["section_key"]
        summaries[section]["files"] += 1
        summaries[section]["groups"].add(row["group_key"] or "Root")
        if row["label_hint"] == "good":
            summaries[section]["good"] += 1
        elif row["label_hint"] == "bad":
            summaries[section]["bad"] += 1

    cards: list[html.Div] = []
    for section in SOURCE_SECTION_NAMES:
        data = summaries[section]
        cards.append(
            html.Div(
                className="summary-card",
                children=[
                    html.Div(SOURCE_SECTION_LABELS[section], className="summary-title"),
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
                f"Pool: {dataset['pooled_files']} | Testing: {dataset['validation_files']}",
                className="run-banner-mode",
            ),
            html.Div(description, className="run-banner-meta"),
        ],
    )


def make_folder_key(section: str, group: str = "") -> str:
    group = (group or "").strip()
    return f"{section}::{group}" if group else section


def parse_folder_key(folder_key: str | None) -> tuple[str | None, str]:
    if not folder_key:
        return None, ""
    section, separator, group = folder_key.partition("::")
    return section or None, group if separator else ""


def list_available_folder_keys(folder_records: list[dict]) -> list[str]:
    section_order = {section: index for index, section in enumerate((*SOURCE_SECTION_NAMES, RUNS_SECTION_KEY))}
    keys: list[str] = []
    for record in sorted(
        folder_records,
        key=lambda item: (
            section_order.get(str(item["section_key"]), 99),
            str(item["group_key"] or "").count("/"),
            str(item["group_key"] or "").lower(),
        ),
    ):
        keys.append(make_folder_key(str(record["section_key"]), str(record["group_key"] or "")))
    return keys


def parent_folder_key(folder_key: str | None) -> str | None:
    section, group = parse_folder_key(folder_key)
    if not section:
        return None
    if not group:
        return None
    if "/" not in group:
        return make_folder_key(section)
    return make_folder_key(section, group.rsplit("/", 1)[0])


def filter_rows_for_folder(rows: list[dict], folder_key: str | None, search_term: str = "") -> list[dict]:
    section, group = parse_folder_key(folder_key)
    filtered = rows
    if section:
        filtered = [row for row in filtered if row["section_key"] == section]
        if group:
            filtered = [row for row in filtered if (row["group_key"] or "") == group]

    query = (search_term or "").strip().lower()
    if query:
        filtered = [
            row
            for row in filtered
            if query in row["name"].lower()
            or query in row["relative_path"].lower()
            or query in row["label_hint"].lower()
            or query in row["group"].lower()
            or query in row["section"].lower()
        ]

    return sorted(filtered, key=lambda row: (row["group"].lower(), row["name"].lower()))


def folder_breadcrumb_parts(folder_key: str | None) -> list[str]:
    section, group = parse_folder_key(folder_key)
    if not section:
        return ["Files"]

    parts = ["Files", SOURCE_SECTION_LABELS.get(section, section.title())]
    if group:
        parts.extend([part for part in group.split("/") if part])
    return parts


def describe_folder_selection(rows: list[dict], folder_key: str | None, search_term: str = "") -> tuple[str, str]:
    visible = filter_rows_for_folder(rows, folder_key, search_term)
    section, group = parse_folder_key(folder_key)

    if not section:
        return "Dataset Contents", f"{len(visible)} item(s) available in this dataset."

    section_label = SOURCE_SECTION_LABELS.get(section, section.title())
    if group:
        title = group
        subtitle = f"{section_label} / {group} | {len(visible)} file(s)"
    else:
        title = section_label
        group_count = len({row['group_key'] or '' for row in rows if row['section_key'] == section})
        subtitle = f"{group_count} folder(s) and {len(visible)} file(s) in this section"

    if (search_term or "").strip():
        subtitle += f" | Search: '{search_term.strip()}'"
    return title, subtitle


def build_browser_rows(rows: list[dict], folder_records: list[dict], folder_key: str | None, run_rows: list[dict] | None = None, search_term: str = "") -> list[dict]:
    section, group = parse_folder_key(folder_key)
    if not section:
        return []

    if section == RUNS_SECTION_KEY:
        browser_rows: list[dict] = []
        visible_runs = list(run_rows or [])
        query = (search_term or "").strip().lower()
        if query:
            visible_runs = [row for row in visible_runs if query in str(row["name"]).lower()]
        for row in visible_runs:
            browser_rows.append(
                {
                    "name": f"[Run] {row['name']}",
                    "size_kb": row["size_kb"],
                    "modified": row["modified"],
                    "row_type": "run",
                    "nav_key": "",
                    "file_id": row["artifact_path"],
                }
            )
        return browser_rows

    prefix = group.strip()
    query = (search_term or "").strip().lower()
    folder_children: dict[str, dict[str, str]] = {}
    file_rows: list[dict] = []

    for folder in folder_records:
        if folder["section_key"] != section:
            continue
        folder_group = str(folder["group_key"] or "")
        if folder_group == prefix:
            continue
        if prefix:
            if not folder_group.startswith(prefix + "/"):
                continue
            remainder = folder_group[len(prefix) + 1 :]
        else:
            remainder = folder_group

        if not remainder or "/" in remainder:
            continue

        folder_children[folder_group] = {
            "name": str(folder["name"]),
            "nav_key": make_folder_key(section, folder_group),
            "modified": str(folder["modified"]),
            "files": int(folder["file_count"]),
        }

    for row in rows:
        if row["section_key"] != section:
            continue
        row_group = row["group_key"] or ""
        if row_group == prefix:
            file_rows.append(row)

    browser_rows: list[dict] = []
    parent_key = parent_folder_key(folder_key)
    if parent_key:
        browser_rows.append(
            {
                "name": "[Back] back to parent folder",
                "size_kb": "",
                "modified": "",
                "row_type": "back",
                "nav_key": parent_key,
                "file_id": "",
            }
        )

    for child in sorted(folder_children.values(), key=lambda item: str(item["name"]).lower()):
        if query and query not in str(child["name"]).lower():
            continue
        browser_rows.append(
            {
                "name": f"[Folder] {child['name']}",
                "size_kb": "",
                "modified": child["modified"],
                "row_type": "folder",
                "nav_key": child["nav_key"],
                "file_id": "",
            }
        )

    visible_files = file_rows
    if query:
        visible_files = [
            row
            for row in file_rows
            if query in row["name"].lower()
            or query in row["relative_path"].lower()
            or query in row["label_hint"].lower()
        ]

    for row in sorted(visible_files, key=lambda item: item["name"].lower()):
        browser_rows.append(
            {
                "name": f"[Audio] {row['name']}",
                "size_kb": row["size_kb"],
                "modified": row["modified"],
                "row_type": "file",
                "nav_key": "",
                "file_id": row["file_id"],
            }
        )

    return browser_rows


def build_file_tree(rows: list[dict], selected_folder_key: str | None = None) -> html.Div:
    if not rows:
        return html.Div("No audio files in this dataset yet.", className="muted-text")

    grouped: dict[str, list[dict]] = {section: [] for section in (*SOURCE_SECTION_NAMES, RUNS_SECTION_KEY)}
    for record in rows:
        grouped.setdefault(str(record["section_key"]), []).append(record)

    section_order = (*SOURCE_SECTION_NAMES, RUNS_SECTION_KEY)
    section_labels = {**SOURCE_SECTION_LABELS, RUNS_SECTION_KEY: RUNS_SECTION_LABEL}

    def render_child_nodes(section: str, section_records: list[dict], parent_group: str) -> list[html.Div]:
        nodes: list[html.Div] = []
        children = sorted(
            [item for item in section_records if item["group_key"] and item["parent_group_key"] == parent_group],
            key=lambda item: str(item["group_key"]).lower(),
        )
        for record in children:
            group_key = make_folder_key(section, str(record["group_key"]))
            grandchildren = render_child_nodes(section, section_records, str(record["group_key"]))
            nodes.append(
                html.Div(
                    className="file-nav-section",
                    children=[
                        html.Button(
                            type="button",
                            id={"type": "file-tree-node", "key": group_key},
                            n_clicks=0,
                            className=f"file-nav-node file-nav-node-child{' is-selected' if selected_folder_key == group_key else ''}",
                            children=[
                                html.Span(className="file-nav-glyph"),
                                html.Span(str(record["name"]), className="file-nav-label"),
                                html.Span(str(record["file_count"]), className="file-nav-count"),
                            ],
                        ),
                        html.Div(grandchildren, className="file-nav-children") if grandchildren else html.Div(),
                    ],
                )
            )
        return nodes

    section_nodes: list[html.Div] = []
    for section in section_order:
        section_records = grouped.get(section, [])
        if not section_records:
            continue

        root_record = next((record for record in section_records if not record["group_key"]), None)
        section_files = int(root_record["file_count"]) if root_record else 0
        section_key = make_folder_key(section)
        child_nodes = render_child_nodes(section, section_records, "")

        section_nodes.append(
            html.Div(
                className="file-nav-section",
                children=[
                    html.Button(
                        type="button",
                        id={"type": "file-tree-node", "key": section_key},
                        n_clicks=0,
                        className=f"file-nav-node file-nav-node-section{' is-selected' if selected_folder_key == section_key else ''}",
                        children=[
                            html.Span(className="file-nav-glyph"),
                            html.Span(section_labels[section], className="file-nav-label"),
                            html.Span(f"{section_files}", className="file-nav-count"),
                        ],
                    ),
                    html.Div(child_nodes, className="file-nav-children"),
                ],
            )
        )

    return html.Div(
        className="file-nav-shell",
        children=[
            html.Div("Dataset items", className="file-nav-heading"),
            html.Div(section_nodes, className="file-nav-tree"),
        ],
    )


def build_attached_run_status(dataset_slug: str, current_run: dict | None = None) -> html.Div:
    dataset = get_dataset_record(dataset_slug)
    artifact_path = (current_run or {}).get("artifact_path")
    run_id = (current_run or {}).get("run_id")
    load_mode = (current_run or {}).get("load_mode")

    if artifact_path:
        mode_text = "Attached saved run ready" if load_mode == "attached_saved" else "Latest session run ready"
        detail_text = run_id or "Saved model bundle selected"
        tone = "ready"
    else:
        mode_text = "No attached run yet"
        detail_text = "Train once on this dataset to cache a reusable model bundle."
        tone = "empty"

    return html.Div(
        className=f"status-badge status-badge-{tone}",
        children=[
            html.Div(dataset["label"], className="status-badge-title"),
            html.Div(mode_text, className="status-badge-mode"),
            html.Div(detail_text, className="status-badge-meta"),
        ],
    )


def build_results_context_status(dataset_slug: str, current_run: dict | None = None) -> html.Div:
    dataset = get_dataset_record(dataset_slug)
    artifact_path = (current_run or {}).get("artifact_path")
    run_id = (current_run or {}).get("run_id")
    load_mode = (current_run or {}).get("load_mode")

    if artifact_path:
        mode_text = "Attached saved run ready" if load_mode == "attached_saved" else "Latest session run ready"
        detail_text = run_id or "Saved model bundle selected"
        tone = "ready"
    else:
        mode_text = "No attached run yet"
        detail_text = "Train once on this dataset to cache a reusable model bundle."
        tone = "empty"

    dataset_summary = (
        f"Files: {dataset['file_count']} | Groups: {dataset['group_count']} | "
        f"Pool: {dataset['pooled_files']} | Testing: {dataset['validation_files']}"
    )

    return html.Div(
        className=f"status-badge status-badge-{tone}",
        children=[
            html.Div(dataset["label"], className="status-badge-title"),
            html.Div(mode_text, className="status-badge-mode"),
            html.Div(dataset_summary, className="status-badge-meta status-badge-summary"),
            html.Div(detail_text, className="status-badge-meta"),
        ],
    )


def build_results_banner(dataset_slug: str, current_run: dict | None = None, bundle: dict | None = None) -> html.Div:
    dataset = get_dataset_record(dataset_slug)
    bundle = bundle or {}
    artifact_path = (current_run or {}).get("artifact_path")

    dataset_summary = (
        f"Files: {dataset['file_count']} | Groups: {dataset['group_count']} | "
        f"Pool: {dataset['pooled_files']} | Testing: {dataset['validation_files']}"
    )

    run_text = (current_run or {}).get("run_id") or bundle.get("run_id") or "Saved model bundle selected"
    if not artifact_path:
        run_text = "Train once on this dataset to cache a reusable model bundle."

    top_text = "Run the pipeline to populate analysis outputs."
    results = bundle.get("results_table", [])
    top_row = results[0] if results else None
    if top_row:
        top_text = (
            f"Best testing accuracy: {top_row['feature_set']} | {top_row['model']} "
            f"= {top_row['validation_accuracy']:.4f}"
        )

    source_text = ""
    if bundle.get("mode") == "loaded-evaluation" and bundle.get("source_run_id"):
        source_text = f" | Evaluated from saved run {bundle['source_run_id']}"

    return html.Div(
        className="run-banner",
        children=[
            html.Div(dataset["label"], className="status-badge-title"),
            html.Div(dataset_summary, className="status-badge-meta status-badge-summary"),
            html.Div(
                f"Current run: {run_text}{source_text}" if artifact_path else run_text,
                className="run-banner-title",
            ),
            html.Div(top_text, className="run-banner-meta"),
        ],
    )


def flatten_diagnostics(bundle: dict) -> list[dict]:
    rows: list[dict] = []
    ordered_keys = [
        "section",
        "group",
        "relative_path",
        "source_file",
        "label",
        "found_hits",
        "expected_hits",
        "status",
        "split_role",
        "delta_used",
        "duration_sec",
        "error",
    ]
    for section in SECTION_NAMES:
        for row in bundle.get("section_diagnostics", {}).get(section, []):
            merged = {
                "section": SECTION_LABELS[section],
                "group": row.get("group", ""),
                "relative_path": row.get("relative_path", ""),
                "source_file": row.get("source_file", ""),
                "label": row.get("label", ""),
                "found_hits": row.get("found_hits", "n/a") if row.get("found_hits", "") != "" else "n/a",
                "expected_hits": row.get("expected_hits", "n/a") if row.get("expected_hits", "") != "" else "n/a",
                "status": row.get("status", ""),
                "split_role": row.get("split_role", section),
                "delta_used": row.get("delta_used", "n/a") if row.get("delta_used", "") != "" else "n/a",
                "duration_sec": row.get("duration_sec", "n/a") if row.get("duration_sec", "") != "" else "n/a",
                "error": row.get("error") or "None",
            }
            rows.append({key: merged.get(key, "") for key in ordered_keys})
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
                                    f"Testing: {result_row.get('validation_accuracy', 0):.4f} | "
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
                                    figure=make_confusion_matrix_figure(matrices["testing"], "Validation Confusion Matrix"),
                                    config={"displayModeBar": False},
                                ),
                                dcc.Graph(
                                    figure=make_confusion_matrix_figure(matrices["validation"], "Testing Confusion Matrix"),
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
