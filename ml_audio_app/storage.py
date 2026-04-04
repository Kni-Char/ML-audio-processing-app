from __future__ import annotations

import base64
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import joblib

from .config import (
    ALLOWED_AUDIO_EXTENSIONS,
    ARTIFACTS_ROOT,
    DATA_ROOT,
    DATASET_META_FILENAME,
    DEFAULT_DATASET_LABEL,
    DEFAULT_DATASET_SLUG,
    DEFAULT_RECORDING_PREFIX,
    EXAMPLE_DATASET_LABEL,
    EXAMPLE_DATASET_SLUG,
    PROCESSED_ROOT,
    RUNS_ROOT,
    SECTION_LABELS,
    SECTION_NAMES,
)


DATASET_LABEL_FALLBACKS = {
    DEFAULT_DATASET_SLUG: DEFAULT_DATASET_LABEL,
    EXAMPLE_DATASET_SLUG: EXAMPLE_DATASET_LABEL,
}


def ensure_app_directories() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_dataset_directories(DEFAULT_DATASET_SLUG)


def sanitize_filename(filename: str, default: str = "audio.wav") -> str:
    clean = Path(filename or "").name.strip()
    clean = clean.replace(" ", "_")
    clean = re.sub(r"[^A-Za-z0-9._ -]", "_", clean)
    clean = clean.strip()
    return clean or default


def slugify_dataset_name(name: str, default: str = DEFAULT_DATASET_SLUG) -> str:
    raw = (name or "").strip().lower()
    raw = raw.replace("_", "-")
    raw = re.sub(r"[^a-z0-9 -]", "", raw)
    raw = re.sub(r"\s+", "-", raw)
    raw = re.sub(r"-{2,}", "-", raw).strip("-")
    return raw or default


def is_allowed_audio_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_AUDIO_EXTENSIONS


def deduplicate_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def decode_data_url(data_url: str) -> bytes:
    if not data_url or "," not in data_url:
        raise ValueError("No audio payload was provided.")
    return base64.b64decode(data_url.split(",", 1)[1])


def label_hint_from_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    if stem.endswith("_g"):
        return "good"
    if stem.endswith("_b"):
        return "bad"
    return "unknown"


def dataset_root(dataset_slug: str) -> Path:
    return DATA_ROOT / slugify_dataset_name(dataset_slug)


def dataset_meta_path(dataset_slug: str) -> Path:
    return dataset_root(dataset_slug) / DATASET_META_FILENAME


def section_dir(dataset_slug: str, section: str) -> Path:
    if section not in SECTION_NAMES:
        raise ValueError(f"Unknown section: {section}")
    return dataset_root(dataset_slug) / section


def normalize_group_path(group: str | None) -> str:
    raw = (group or "").strip().replace("\\", "/")
    if not raw:
        return ""

    parts: list[str] = []
    for part in raw.split("/"):
        part = part.strip()
        if not part or part in {".", ".."}:
            continue
        parts.append(sanitize_filename(part, default="group"))
    return "/".join(parts)


def sanitize_relative_audio_path(relative_path: str) -> str:
    raw = str(PurePosixPath(relative_path.replace("\\", "/")))
    parts = [part for part in raw.split("/") if part not in {"", ".", ".."}]
    if not parts:
        raise ValueError("A relative audio path is required.")

    filename = sanitize_filename(parts[-1])
    groups = normalize_group_path("/".join(parts[:-1]))
    return str(PurePosixPath(groups, filename)) if groups else filename


def ensure_dataset_directories(
    dataset_slug: str,
    label: str | None = None,
    description: str | None = None,
) -> None:
    root = dataset_root(dataset_slug)
    root.mkdir(parents=True, exist_ok=True)
    for section in SECTION_NAMES:
        section_dir(dataset_slug, section).mkdir(parents=True, exist_ok=True)

    meta_file = dataset_meta_path(dataset_slug)
    if label is None and meta_file.exists():
        return

    meta = read_dataset_metadata(dataset_slug)
    meta.update(
        {
            "slug": slugify_dataset_name(dataset_slug),
            "label": label or meta.get("label") or DATASET_LABEL_FALLBACKS.get(dataset_slug, humanize_dataset_slug(dataset_slug)),
            "description": description if description is not None else meta.get("description", ""),
        }
    )
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def humanize_dataset_slug(dataset_slug: str) -> str:
    return slugify_dataset_name(dataset_slug).replace("-", " ").title()


def read_dataset_metadata(dataset_slug: str) -> dict:
    meta_file = dataset_meta_path(dataset_slug)
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            pass
    return {
        "slug": slugify_dataset_name(dataset_slug),
        "label": DATASET_LABEL_FALLBACKS.get(dataset_slug, humanize_dataset_slug(dataset_slug)),
        "description": "",
    }


def get_dataset_label(dataset_slug: str) -> str:
    return read_dataset_metadata(dataset_slug).get("label", humanize_dataset_slug(dataset_slug))


def get_default_dataset_slug() -> str:
    example_root = dataset_root(EXAMPLE_DATASET_SLUG)
    if example_root.exists():
        return EXAMPLE_DATASET_SLUG
    return DEFAULT_DATASET_SLUG


def list_dataset_records() -> list[dict]:
    ensure_app_directories()
    records: list[dict] = []

    roots = []
    for path in DATA_ROOT.iterdir():
        if not path.is_dir():
            continue
        if any((path / section).exists() for section in SECTION_NAMES):
            roots.append(path)

    if dataset_root(DEFAULT_DATASET_SLUG) not in roots:
        ensure_dataset_directories(DEFAULT_DATASET_SLUG)
        roots.append(dataset_root(DEFAULT_DATASET_SLUG))

    def sort_key(path: Path) -> tuple[int, str]:
        slug = path.name
        preferred = 0 if slug == get_default_dataset_slug() else 1
        return preferred, slug

    for path in sorted(roots, key=sort_key):
        dataset_slug = path.name
        meta = read_dataset_metadata(dataset_slug)
        rows = list_audio_files(dataset_slug)
        records.append(
            {
                "slug": dataset_slug,
                "label": meta.get("label", humanize_dataset_slug(dataset_slug)),
                "description": meta.get("description", ""),
                "file_count": len(rows),
                "group_count": len({(row["section_key"], row["group_key"]) for row in rows}),
                "training_files": sum(1 for row in rows if row["section_key"] == "training"),
                "testing_files": sum(1 for row in rows if row["section_key"] == "testing"),
                "validation_files": sum(1 for row in rows if row["section_key"] == "validation"),
            }
        )

    return records


def list_dataset_options() -> list[dict[str, str]]:
    return [{"label": record["label"], "value": record["slug"]} for record in list_dataset_records()]


def get_dataset_record(dataset_slug: str) -> dict:
    dataset_slug = slugify_dataset_name(dataset_slug)
    for record in list_dataset_records():
        if record["slug"] == dataset_slug:
            return record
    ensure_dataset_directories(dataset_slug)
    return next(record for record in list_dataset_records() if record["slug"] == dataset_slug)


def make_file_id(dataset_slug: str, section: str, relative_path: str) -> str:
    return f"{slugify_dataset_name(dataset_slug)}|{section}|{sanitize_relative_audio_path(relative_path)}"


def split_file_id(file_id: str) -> tuple[str, str, str]:
    dataset_slug, section, relative_path = file_id.split("|", 2)
    if section not in SECTION_NAMES:
        raise ValueError(f"Unknown section in file id: {section}")
    return slugify_dataset_name(dataset_slug), section, sanitize_relative_audio_path(relative_path)


def list_audio_paths(dataset_slug: str, section: str) -> list[Path]:
    folder = section_dir(dataset_slug, section)
    if not folder.exists():
        return []
    return sorted(
        [
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS
        ],
        key=lambda item: str(item.relative_to(folder)).lower(),
    )


def list_audio_files(dataset_slug: str) -> list[dict[str, str]]:
    ensure_dataset_directories(dataset_slug)
    rows: list[dict[str, str]] = []

    for section in SECTION_NAMES:
        root = section_dir(dataset_slug, section)
        for path in list_audio_paths(dataset_slug, section):
            stats = path.stat()
            relative_path = path.relative_to(root).as_posix()
            group_key = "" if path.parent == root else path.parent.relative_to(root).as_posix()
            rows.append(
                {
                    "file_id": make_file_id(dataset_slug, section, relative_path),
                    "dataset_slug": dataset_slug,
                    "dataset_label": get_dataset_label(dataset_slug),
                    "section": SECTION_LABELS[section],
                    "section_key": section,
                    "group": group_key or "Root",
                    "group_key": group_key,
                    "name": path.name,
                    "relative_path": relative_path,
                    "label_hint": label_hint_from_name(path.name),
                    "size_kb": f"{stats.st_size / 1024:.1f}",
                    "modified": datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    return rows


def list_group_summary_rows(dataset_slug: str) -> list[dict[str, str | int]]:
    rows = list_audio_files(dataset_slug)
    grouped: dict[tuple[str, str], dict[str, str | int]] = {}

    for row in rows:
        key = (row["section_key"], row["group_key"])
        if key not in grouped:
            grouped[key] = {
                "section": row["section"],
                "group": row["group"],
                "files": 0,
                "good_files": 0,
                "bad_files": 0,
            }
        grouped[key]["files"] += 1
        if row["label_hint"] == "good":
            grouped[key]["good_files"] += 1
        elif row["label_hint"] == "bad":
            grouped[key]["bad_files"] += 1

    section_index = {SECTION_LABELS[section]: index for index, section in enumerate(SECTION_NAMES)}
    return sorted(grouped.values(), key=lambda item: (section_index.get(str(item["section"]), 99), str(item["group"]).lower()))


def save_uploaded_files(
    contents_list: Sequence[str] | None,
    filenames: Sequence[str] | None,
    dataset_slug: str,
    section: str,
    group: str = "",
) -> dict[str, list[str]]:
    ensure_dataset_directories(dataset_slug)
    if section not in SECTION_NAMES:
        raise ValueError(f"Unknown section: {section}")
    if not contents_list or not filenames:
        return {"saved": [], "skipped": []}

    saved: list[str] = []
    skipped: list[str] = []
    group_path = normalize_group_path(group)
    target_dir = section_dir(dataset_slug, section) / Path(group_path) if group_path else section_dir(dataset_slug, section)
    target_dir.mkdir(parents=True, exist_ok=True)

    for contents, filename in zip(contents_list, filenames):
        if not is_allowed_audio_file(filename):
            skipped.append(filename)
            continue

        clean_name = sanitize_filename(filename)
        destination = deduplicate_path(target_dir / clean_name)
        destination.write_bytes(decode_data_url(contents))
        saved.append(destination.name)

    return {"saved": saved, "skipped": skipped}


def build_recording_name(
    prefix: str = DEFAULT_RECORDING_PREFIX,
    sample_number: int = 1,
    condition: str = "g",
    extension: str = ".wav",
) -> str:
    clean_prefix = sanitize_filename(prefix or DEFAULT_RECORDING_PREFIX, default=DEFAULT_RECORDING_PREFIX)
    clean_prefix = clean_prefix.replace(".", "")
    clean_condition = "g" if str(condition).lower().startswith("g") else "b"
    return f"{clean_prefix}_{int(sample_number)}_{clean_condition}{extension}"


def save_recording(
    data_url: str,
    dataset_slug: str,
    section: str,
    group: str,
    prefix: str,
    sample_number: int,
    condition: str,
) -> Path:
    ensure_dataset_directories(dataset_slug)
    if section not in SECTION_NAMES:
        raise ValueError(f"Unknown section: {section}")

    filename = build_recording_name(prefix=prefix, sample_number=sample_number, condition=condition)
    group_path = normalize_group_path(group)
    target_dir = section_dir(dataset_slug, section) / Path(group_path) if group_path else section_dir(dataset_slug, section)
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = deduplicate_path(target_dir / filename)
    destination.write_bytes(decode_data_url(data_url))
    return destination


def resolve_audio_path(dataset_slug: str, section: str, relative_path: str) -> Path:
    if section not in SECTION_NAMES:
        raise ValueError(f"Unknown section: {section}")
    safe_relative = sanitize_relative_audio_path(relative_path)
    return section_dir(dataset_slug, section) / Path(PurePosixPath(safe_relative))


def move_files(
    file_ids: Iterable[str],
    target_section: str,
    target_group: str = "",
    target_dataset_slug: str | None = None,
) -> list[str]:
    if target_section not in SECTION_NAMES:
        raise ValueError(f"Unknown target section: {target_section}")

    moved: list[str] = []
    group_path = normalize_group_path(target_group)

    for file_id in file_ids:
        dataset_slug, section, relative_path = split_file_id(file_id)
        dataset_target = slugify_dataset_name(target_dataset_slug or dataset_slug)
        ensure_dataset_directories(dataset_target)
        target_dir = section_dir(dataset_target, target_section) / Path(group_path) if group_path else section_dir(dataset_target, target_section)
        target_dir.mkdir(parents=True, exist_ok=True)
        source = resolve_audio_path(dataset_slug, section, relative_path)
        if not source.exists():
            continue
        destination = deduplicate_path(target_dir / source.name)
        source.rename(destination)
        moved.append(destination.name)

    return moved


def _cleanup_empty_parents(path: Path, stop_at: Path) -> None:
    current = path.parent
    while current != stop_at and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def delete_files(file_ids: Iterable[str]) -> list[str]:
    deleted: list[str] = []
    for file_id in file_ids:
        dataset_slug, section, relative_path = split_file_id(file_id)
        path = resolve_audio_path(dataset_slug, section, relative_path)
        if path.exists():
            path.unlink()
            deleted.append(path.name)
            _cleanup_empty_parents(path, section_dir(dataset_slug, section))
    return deleted


def save_run_artifact(bundle: dict) -> Path:
    ensure_app_directories()
    run_id = bundle["run_id"]
    destination = RUNS_ROOT / f"{run_id}.joblib"
    joblib.dump(bundle, destination)
    return destination


def load_run_artifact(run_path: str | Path) -> dict:
    return joblib.load(Path(run_path))


def list_run_artifacts() -> list[dict[str, str]]:
    ensure_app_directories()
    options: list[dict[str, str]] = []
    for path in sorted(RUNS_ROOT.glob("*.joblib"), key=lambda item: item.stat().st_mtime, reverse=True):
        modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        label = path.stem
        try:
            bundle = load_run_artifact(path)
            dataset_label = bundle.get("dataset_label")
            if dataset_label:
                label = f"{path.stem} | {dataset_label}"
        except Exception:
            pass
        options.append({"label": f"{label} ({modified})", "value": str(path)})
    return options
