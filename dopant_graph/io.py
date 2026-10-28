"""Input discovery and label parsing for initial POSCAR datasets."""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dopant_graph.geometry import infer_supercell_from_path


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    poscar_path: Path
    labels: dict[str, float]
    supercell: str | None = None
    dopant_type: str | None = None
    source_dir: Path | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "poscar_path": str(self.poscar_path),
            "supercell": self.supercell or "",
            "dopant_type": self.dopant_type or "",
            "source_dir": str(self.source_dir or self.poscar_path.parent),
            **{f"label_{key}": value for key, value in self.labels.items()},
        }


def parse_label_names(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in value if str(item).strip()]
    if not names:
        raise ValueError("At least one label name is required.")
    return names


def resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def read_key_value_text(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    first_number: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if first_number is None:
            match = re.search(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?", line)
            if match:
                first_number = match.group(0)
        if "=" in line:
            key, value = [part.strip() for part in line.split("=", 1)]
        elif ":" in line:
            key, value = [part.strip() for part in line.split(":", 1)]
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            key, value = parts[0], parts[1]
        if key:
            values[key] = value
    if first_number is not None:
        values.setdefault("__first_number__", first_number)
    return values


def read_label_file(path: Path, label_names: list[str]) -> dict[str, float]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {name: float(payload[name]) for name in label_names}
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            row = next(reader)
        return {name: float(row[name]) for name in label_names}

    values = read_key_value_text(path)
    labels: dict[str, float] = {}
    for name in label_names:
        if name in values:
            labels[name] = float(values[name])
        elif len(label_names) == 1 and "__first_number__" in values:
            labels[name] = float(values["__first_number__"])
        else:
            raise KeyError(f"Could not find label {name!r} in {path}.")
    return labels


def read_manifest(
    manifest_path: Path,
    label_names: list[str],
    poscar_column: str = "poscar_path",
) -> list[SampleRecord]:
    manifest_path = manifest_path.resolve()
    records: list[SampleRecord] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest {manifest_path} has no header.")
        missing = [name for name in [poscar_column, *label_names] if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Manifest {manifest_path} is missing columns: {', '.join(missing)}")
        for row_index, row in enumerate(reader, start=2):
            poscar_path = resolve_path(row[poscar_column], manifest_path.parent)
            sample_id = row.get("sample_id") or row.get("id") or poscar_path.parent.name or f"row_{row_index}"
            labels = {name: float(row[name]) for name in label_names}
            supercell = row.get("supercell") or infer_supercell_from_path(poscar_path.parts)
            records.append(
                SampleRecord(
                    sample_id=str(sample_id),
                    poscar_path=poscar_path,
                    labels=labels,
                    supercell=supercell,
                    dopant_type=row.get("dopant_type") or row.get("category"),
                    source_dir=poscar_path.parent,
                )
            )
    return records


def discover_records(
    data_root: Path,
    label_names: list[str],
    poscar_names: list[str],
    label_file: str,
) -> tuple[list[SampleRecord], list[dict[str, str]]]:
    data_root = data_root.resolve()
    poscar_names_set = set(poscar_names)
    records: list[SampleRecord] = []
    skipped: list[dict[str, str]] = []
    matched_poscar_count = 0

    if not data_root.is_dir():
        return [], [
            {
                "sample_id": "",
                "poscar_path": "",
                "reason": f"data_root does not exist or is not a directory: {data_root}",
            }
        ]

    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.name not in poscar_names_set:
            continue
        matched_poscar_count += 1
        sample_dir = path.parent
        label_path = sample_dir / label_file
        if not label_path.is_file():
            skipped.append(
                {
                    "sample_id": sample_dir.name,
                    "poscar_path": str(path),
                    "reason": f"missing label file {label_file}",
                }
            )
            continue
        try:
            labels = read_label_file(label_path, label_names)
        except Exception as exc:
            skipped.append(
                {
                    "sample_id": sample_dir.name,
                    "poscar_path": str(path),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        records.append(
            SampleRecord(
                sample_id=sample_dir.name,
                poscar_path=path.resolve(),
                labels=labels,
                supercell=infer_supercell_from_path(path.parts),
                dopant_type=sample_dir.parent.parent.name if len(sample_dir.parents) >= 2 else None,
                source_dir=sample_dir,
            )
        )
    if matched_poscar_count == 0:
        skipped.append(
            {
                "sample_id": "",
                "poscar_path": "",
                "reason": (
                    "no POSCAR files matched poscar_names="
                    f"{','.join(poscar_names)} under data_root={data_root}"
                ),
            }
        )
    return records, skipped


def split_records(
    records: list[SampleRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[SampleRecord]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}.")

    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    n_train = int(round(total * train_ratio))
    n_val = int(round(total * val_ratio))
    if n_train + n_val > total:
        n_val = max(0, total - n_train)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
