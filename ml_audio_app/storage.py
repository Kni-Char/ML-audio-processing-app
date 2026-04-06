from __future__ import annotations

import base64
import json
import re
import shutil
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
    SOURCE_SECTION_ALIASES,
    SOURCE_SECTION_LABELS,
    SOURCE_SECTION_NAMES,
)

RUNS_SECTION_KEY = "runs"
RUNS_SECTION_LABEL = "Saved Runs"


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


def deduplicate_directory(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.name
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}")
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
    section = canonical_source_section(section)
    if section not in SOURCE_SECTION_NAMES:
        raise ValueError(f"Unknown section: {section}")
    return dataset_root(dataset_slug) / section


def canonical_source_section(section: str) -> str:
    normalized = slugify_dataset_name(section, default="")
    mapped = SOURCE_SECTION_ALIASES.get(normalized)
    if not mapped:
        raise ValueError(f"Unknown section: {section}")
    return mapped


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
    for section in SOURCE_SECTION_NAMES:
        section_dir(dataset_slug, section).mkdir(parents=True, exist_ok=True)

    migrate_legacy_train_test_sections(dataset_slug)

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


def migrate_legacy_train_test_sections(dataset_slug: str) -> None:
    root = dataset_root(dataset_slug)
    pooled_root = root / "pooled"
    pooled_root.mkdir(parents=True, exist_ok=True)

    for legacy_name in ("training", "testing"):
        legacy_root = root / legacy_name
        if not legacy_root.exists():
            continue

        files = [path for path in legacy_root.rglob("*") if path.is_file() and path.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS]
        for file_path in files:
            relative = file_path.relative_to(legacy_root)
            destination = deduplicate_path(pooled_root / relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            file_path.rename(destination)

        for empty_dir in sorted(
            [path for path in legacy_root.rglob("*") if path.is_dir()],
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            try:
                empty_dir.rmdir()
            except OSError:
                pass
        try:
            legacy_root.rmdir()
        except OSError:
            pass


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
        if any((path / section).exists() for section in ("pooled", "validation", "training", "testing")):
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
                "pooled_files": sum(1 for row in rows if row["section_key"] == "pooled"),
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
    return {
        "slug": dataset_slug,
        "label": DATASET_LABEL_FALLBACKS.get(dataset_slug, humanize_dataset_slug(dataset_slug)),
        "description": "",
        "file_count": 0,
        "group_count": 0,
        "pooled_files": 0,
        "validation_files": 0,
    }


def resolve_existing_dataset_slug(name_or_slug: str | None) -> str | None:
    raw = (name_or_slug or "").strip()
    if not raw:
        return None

    slug_candidate = slugify_dataset_name(raw)
    records = list_dataset_records()
    for record in records:
        if record["slug"] == slug_candidate:
            return record["slug"]

    lower_label = raw.lower()
    for record in records:
        if str(record["label"]).strip().lower() == lower_label:
            return record["slug"]

    return None


def _deduplicate_dataset_slug(preferred_slug: str) -> str:
    base_slug = slugify_dataset_name(preferred_slug)
    candidate = base_slug
    counter = 2
    while dataset_root(candidate).exists():
        candidate = f"{base_slug}-{counter}"
        counter += 1
    return candidate


def clone_dataset_bundle(
    source_dataset_slug: str,
    target_label: str,
    target_description: str = "",
) -> dict[str, str]:
    source_slug = slugify_dataset_name(source_dataset_slug)
    source_root = dataset_root(source_slug)
    if not source_root.exists():
        raise ValueError(f"Source dataset '{source_slug}' does not exist.")

    clean_label = (target_label or "").strip()
    if not clean_label:
        raise ValueError("Enter a dataset bundle name before saving.")

    target_slug = _deduplicate_dataset_slug(clean_label)
    target_root = dataset_root(target_slug)
    shutil.copytree(source_root, target_root)

    source_meta = read_dataset_metadata(source_slug)
    ensure_dataset_directories(
        target_slug,
        label=clean_label,
        description=(target_description or "").strip() or source_meta.get("description", ""),
    )
    return {"slug": target_slug, "label": clean_label}


def delete_dataset_bundle(dataset_slug: str) -> dict[str, str]:
    source_slug = slugify_dataset_name(dataset_slug)
    if source_slug in {DEFAULT_DATASET_SLUG, EXAMPLE_DATASET_SLUG}:
        raise ValueError("Protected dataset bundles cannot be deleted.")

    source_root = dataset_root(source_slug)
    if not source_root.exists():
        raise ValueError(f"Dataset bundle '{source_slug}' does not exist.")

    bundle_label = get_dataset_label(source_slug)
    shutil.rmtree(source_root)

    fallback_slug = get_default_dataset_slug()
    ensure_dataset_directories(fallback_slug)
    return {"slug": fallback_slug, "label": bundle_label}


def make_file_id(dataset_slug: str, section: str, relative_path: str) -> str:
    return f"{slugify_dataset_name(dataset_slug)}|{canonical_source_section(section)}|{sanitize_relative_audio_path(relative_path)}"


def split_file_id(file_id: str) -> tuple[str, str, str]:
    dataset_slug, section, relative_path = file_id.split("|", 2)
    return slugify_dataset_name(dataset_slug), canonical_source_section(section), sanitize_relative_audio_path(relative_path)


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
    if not dataset_root(dataset_slug).exists():
        return []
    rows: list[dict[str, str]] = []

    for section in SOURCE_SECTION_NAMES:
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
                    "section": SOURCE_SECTION_LABELS[section],
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


def list_folder_records(dataset_slug: str) -> list[dict[str, str | int | bool]]:
    if not dataset_root(dataset_slug).exists():
        return []
    audio_rows = list_audio_files(dataset_slug)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    modified_lookup: dict[tuple[str, str], str] = {}

    for row in audio_rows:
        section = str(row["section_key"])
        group = str(row["group_key"] or "")
        modified = str(row["modified"])
        counts[(section, "")] += 1
        modified_lookup[(section, "")] = max(modified_lookup.get((section, ""), ""), modified)

        if group:
            parts = group.split("/")
            for index in range(len(parts)):
                ancestor = "/".join(parts[: index + 1])
                counts[(section, ancestor)] += 1
                modified_lookup[(section, ancestor)] = max(modified_lookup.get((section, ancestor), ""), modified)

    records: list[dict[str, str | int | bool]] = []
    for section in SOURCE_SECTION_NAMES:
        root = section_dir(dataset_slug, section)
        directories = [root] + sorted([path for path in root.rglob("*") if path.is_dir()], key=lambda item: item.as_posix().lower())
        for directory in directories:
            group = "" if directory == root else directory.relative_to(root).as_posix()
            parent_group = ""
            if group and "/" in group:
                parent_group = group.rsplit("/", 1)[0]
            name = SOURCE_SECTION_LABELS[section] if not group else directory.name
            folder_modified = datetime.fromtimestamp(directory.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            modified = max(modified_lookup.get((section, group), ""), folder_modified)
            records.append(
                {
                    "section_key": section,
                    "section_label": SOURCE_SECTION_LABELS[section],
                    "group_key": group,
                    "parent_group_key": parent_group,
                    "name": name,
                    "file_count": int(counts.get((section, group), 0)),
                    "modified": modified,
                    "is_root": group == "",
                }
            )

    run_rows = list_run_file_rows(dataset_slug)
    latest_modified = max((str(row["modified"]) for row in run_rows), default="")
    records.append(
        {
            "section_key": RUNS_SECTION_KEY,
            "section_label": RUNS_SECTION_LABEL,
            "group_key": "",
            "parent_group_key": "",
            "name": RUNS_SECTION_LABEL,
            "file_count": len(run_rows),
            "modified": latest_modified,
            "is_root": True,
        }
    )

    return records


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

    section_index = {SOURCE_SECTION_LABELS[section]: index for index, section in enumerate(SOURCE_SECTION_NAMES)}
    return sorted(grouped.values(), key=lambda item: (section_index.get(str(item["section"]), 99), str(item["group"]).lower()))


def save_uploaded_files(
    contents_list: Sequence[str] | None,
    filenames: Sequence[str] | None,
    dataset_slug: str,
    section: str,
    group: str = "",
    relative_paths: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    ensure_dataset_directories(dataset_slug)
    section = canonical_source_section(section)
    if not contents_list or not filenames:
        return {"saved": [], "skipped": []}

    normalized_contents = [contents_list] if isinstance(contents_list, str) else list(contents_list)
    normalized_filenames = [filenames] if isinstance(filenames, str) else list(filenames)
    if not normalized_contents or not normalized_filenames:
        return {"saved": [], "skipped": []}

    saved: list[str] = []
    skipped: list[str] = []
    group_path = normalize_group_path(group)
    target_dir = section_dir(dataset_slug, section) / Path(group_path) if group_path else section_dir(dataset_slug, section)
    target_dir.mkdir(parents=True, exist_ok=True)
    normalized_relative_paths = list(relative_paths or [])

    # When uploading into a root section, allow the user to pick one parent folder
    # that contains multiple child folders such as Set1/Set2/... and strip that
    # wrapper folder so the child folders land directly in the selected section.
    if not group_path and normalized_relative_paths:
        parsed_parts = [
            PurePosixPath(sanitize_relative_audio_path(path)).parts
            for path in normalized_relative_paths
            if path and any(sep in path for sep in ["/", "\\"])
        ]
        if parsed_parts and all(len(parts) >= 3 for parts in parsed_parts):
            second_level_names = {parts[1] for parts in parsed_parts}
            if len(second_level_names) > 1:
                normalized_relative_paths = [
                    str(PurePosixPath(*PurePosixPath(sanitize_relative_audio_path(path)).parts[1:]))
                    if path and any(sep in path for sep in ["/", "\\"])
                    else path
                    for path in normalized_relative_paths
                ]

    for index, (contents, filename) in enumerate(zip(normalized_contents, normalized_filenames)):
        if not is_allowed_audio_file(filename):
            skipped.append(filename)
            continue

        relative_hint = normalized_relative_paths[index] if normalized_relative_paths and index < len(normalized_relative_paths) else ""
        if group_path:
            clean_relative = sanitize_filename(filename)
            destination = deduplicate_path(target_dir / clean_relative)
        else:
            normalized_relative = sanitize_relative_audio_path(relative_hint) if relative_hint and any(sep in relative_hint for sep in ["/", "\\"]) else sanitize_filename(filename)
            destination = deduplicate_path(target_dir / Path(PurePosixPath(normalized_relative)))

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(decode_data_url(contents))
        saved.append(destination.relative_to(target_dir).as_posix())

    return {"saved": saved, "skipped": skipped}


def create_subfolder(
    dataset_slug: str,
    section: str,
    parent_group: str = "",
    folder_name: str = "",
) -> Path:
    ensure_dataset_directories(dataset_slug)
    section = canonical_source_section(section)
    parent_group_path = normalize_group_path(parent_group)
    clean_folder_name = sanitize_filename(folder_name or "NewFolder", default="NewFolder").replace(".", "_")
    parent_dir = section_dir(dataset_slug, section) / Path(parent_group_path) if parent_group_path else section_dir(dataset_slug, section)
    parent_dir.mkdir(parents=True, exist_ok=True)
    destination = deduplicate_directory(parent_dir / clean_folder_name)
    destination.mkdir(parents=True, exist_ok=False)
    return destination


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
    section = canonical_source_section(section)

    filename = build_recording_name(prefix=prefix, sample_number=sample_number, condition=condition)
    group_path = normalize_group_path(group)
    target_dir = section_dir(dataset_slug, section) / Path(group_path) if group_path else section_dir(dataset_slug, section)
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = deduplicate_path(target_dir / filename)
    destination.write_bytes(decode_data_url(data_url))
    return destination


def resolve_audio_path(dataset_slug: str, section: str, relative_path: str) -> Path:
    section = canonical_source_section(section)
    safe_relative = sanitize_relative_audio_path(relative_path)
    return section_dir(dataset_slug, section) / Path(PurePosixPath(safe_relative))


def move_files(
    file_ids: Iterable[str],
    target_section: str,
    target_group: str = "",
    target_dataset_slug: str | None = None,
) -> list[str]:
    target_section = canonical_source_section(target_section)

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


def delete_folders(dataset_slug: str, folder_keys: Iterable[tuple[str, str]]) -> list[str]:
    deleted: list[str] = []
    normalized_targets: list[tuple[str, str]] = []

    for section, group in folder_keys:
        safe_section = canonical_source_section(section)
        safe_group = normalize_group_path(group)
        if not safe_group:
            continue
        normalized_targets.append((safe_section, safe_group))

    unique_targets = sorted(set(normalized_targets), key=lambda item: (item[0], len(item[1]), item[1].lower()))
    pruned_targets: list[tuple[str, str]] = []
    for section, group in unique_targets:
        if any(
            section == existing_section and (group == existing_group or group.startswith(f"{existing_group}/"))
            for existing_section, existing_group in pruned_targets
        ):
            continue
        pruned_targets.append((section, group))

    for section, group in pruned_targets:
        root = section_dir(dataset_slug, section)
        target = root / Path(group)
        if not target.exists() or not target.is_dir():
            continue
        shutil.rmtree(target)
        deleted.append(target.name)

    return deleted


def save_run_artifact(bundle: dict) -> Path:
    ensure_app_directories()
    run_id = bundle["run_id"]
    destination = RUNS_ROOT / f"{run_id}.joblib"
    joblib.dump(bundle, destination)
    return destination


def load_run_artifact(run_path: str | Path) -> dict:
    return joblib.load(Path(run_path))


def _run_artifact_record(path: Path) -> dict[str, str] | None:
    modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    label = path.stem
    dataset_slug = ""
    try:
        bundle = load_run_artifact(path)
        dataset_label = bundle.get("dataset_label")
        dataset_slug = str(bundle.get("dataset_slug", "") or "")
        if dataset_label:
            label = f"{path.stem} | {dataset_label}"
    except Exception:
        pass

    return {
        "label": f"{label} ({modified})",
        "value": str(path),
        "dataset_slug": dataset_slug,
        "modified": modified,
    }


def list_run_file_rows(dataset_slug: str | None = None) -> list[dict[str, str]]:
    ensure_app_directories()
    normalized_dataset = slugify_dataset_name(dataset_slug) if dataset_slug else None
    rows: list[dict[str, str]] = []

    for path in sorted(RUNS_ROOT.glob("*.joblib"), key=lambda item: item.stat().st_mtime, reverse=True):
        record = _run_artifact_record(path)
        if not record:
            continue
        if normalized_dataset and record.get("dataset_slug") != normalized_dataset:
            continue
        rows.append(
            {
                "name": path.name,
                "artifact_path": str(path),
                "size_kb": f"{path.stat().st_size / 1024:.1f}",
                "modified": record.get("modified", ""),
                "dataset_slug": str(record.get("dataset_slug", "")),
            }
        )

    return rows


def list_run_artifacts(dataset_slug: str | None = None) -> list[dict[str, str]]:
    ensure_app_directories()
    options: list[dict[str, str]] = []
    normalized_dataset = slugify_dataset_name(dataset_slug) if dataset_slug else None

    for path in sorted(RUNS_ROOT.glob("*.joblib"), key=lambda item: item.stat().st_mtime, reverse=True):
        record = _run_artifact_record(path)
        if not record:
            continue
        if normalized_dataset and record.get("dataset_slug") != normalized_dataset:
            continue
        options.append({"label": record["label"], "value": record["value"]})

    return options


def get_latest_run_artifact(dataset_slug: str | None = None) -> dict[str, str] | None:
    records = list_run_artifacts(dataset_slug)
    if not records:
        return None
    return records[0]


def delete_run_artifacts(run_paths: Iterable[str]) -> list[str]:
    deleted: list[str] = []
    for run_path in run_paths:
        path = Path(run_path)
        if path.exists() and path.is_file():
            path.unlink()
            deleted.append(path.name)
    return deleted
