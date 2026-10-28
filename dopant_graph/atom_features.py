"""Load per-element properties from the bundled configuration files."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable


DEFAULT_ATOM_FEATURE_DIR = Path(__file__).resolve().parents[1] / "hdff_model_portable" / "config"


def canonical_property_name(name: str) -> str:
    value = name.strip().lower().replace("-", "_")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    if not value:
        raise ValueError(f"Invalid empty atomic-property name derived from {name!r}.")
    return value


def _read_one(path: Path) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected 'property = value'.")
        raw_name, raw_value = (part.strip() for part in line.split("=", 1))
        name = canonical_property_name(raw_name)
        if name in names:
            raise ValueError(f"{path}:{line_number}: duplicated property {name!r}.")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric value {raw_value!r}.") from exc
        names.append(name)
        values.append(value)
    if not names:
        raise ValueError(f"Atomic-property file is empty: {path}")
    return names, values


def load_atom_feature_table(
    feature_dir: str | Path = DEFAULT_ATOM_FEATURE_DIR,
    required_elements: Iterable[str] | None = None,
) -> tuple[tuple[str, ...], dict[str, tuple[float, ...]]]:
    root = Path(feature_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Atomic-property directory does not exist: {root}")
    paths = sorted(root.glob("*.txt"), key=lambda item: item.stem)
    if required_elements is not None:
        expected = tuple(required_elements)
        missing = [symbol for symbol in expected if not (root / f"{symbol}.txt").is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing atomic-property files in {root}: {', '.join(missing)}"
            )
        paths = [root / f"{symbol}.txt" for symbol in expected]
    if not paths:
        raise FileNotFoundError(f"No element .txt files were found in {root}")

    schema: list[str] | None = None
    table: dict[str, tuple[float, ...]] = {}
    for path in paths:
        names, values = _read_one(path)
        if schema is None:
            schema = names
        elif names != schema:
            raise ValueError(
                f"Atomic-property schema/order in {path} differs from the first element file."
            )
        table[path.stem] = tuple(values)
    assert schema is not None
    return tuple(schema), table


def atom_feature_source_sha256(feature_dir: str | Path = DEFAULT_ATOM_FEATURE_DIR) -> str:
    root = Path(feature_dir).expanduser().resolve()
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.txt"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

