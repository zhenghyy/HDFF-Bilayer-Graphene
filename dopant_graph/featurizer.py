"""Descriptor construction for the HDFF-Net dopant-site model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dopant_graph.constants import (
    ATOM_PROPERTIES,
    ATOM_PROPERTY_NAMES,
    BASE_ELEMENT,
    FEATURE_VERSION,
    LAYER_RELATIONS,
    LEGACY_OP_DOPANT_ELEMENTS,
    LEGACY_OP_ELEMENTS,
    LEGACY_OP_GLOBAL_FEATURE_NAMES,
    N_LOCAL_SHELLS,
    SHELL_DISTANCE_TOLERANCE,
    SHELL_RANGES,
    SUPPORTED_DOPANT_ELEMENTS,
    WINDOW_EDGE_ORDER,
    atomic_number,
    element_mapping_for,
    element_symbols_for,
    global_feature_names_for,
    is_legacy_op_dopant_set,
    sort_dopant_elements,
)
from dopant_graph.geometry import (
    assign_bilayer_indices,
    find_equal_distance_shell_neighbors,
    find_projection_atom,
    infer_supercell_dims,
    special_pair_distances,
)
from dopant_graph.io import SampleRecord
from dopant_graph.structure import read_structure


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: list[list[float]] | np.ndarray, dim: int | None = None) -> "StandardScaler":
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            if dim is None:
                raise ValueError("Cannot fit an empty scaler without dim.")
            return cls(np.zeros(dim, dtype=np.float64), np.ones(dim, dtype=np.float64))
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    def transform(self, values: list[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return ((array - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: list[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return (array * self.std + self.mean).astype(np.float32)

    def to_dict(self, names: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        if names is not None:
            payload["names"] = list(names)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StandardScaler":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
        )


@dataclass
class FeatureOptions:
    n_shells: int = N_LOCAL_SHELLS
    shell_tolerance: float = SHELL_DISTANCE_TOLERANCE
    image_range: int = 1
    atom_property_names: tuple[str, ...] | list[str] = ATOM_PROPERTY_NAMES
    element_shell_counts: dict[str, int] = field(default_factory=dict)
    dopant_elements: tuple[str, ...] | list[str] | None = None
    atom_property_table: dict[str, tuple[float, ...] | list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n_shells = int(self.n_shells)
        self.shell_tolerance = float(self.shell_tolerance)
        self.image_range = int(self.image_range)
        if self.n_shells < 1:
            raise ValueError("n_shells must be >= 1.")
        if self.shell_tolerance <= 0:
            raise ValueError("shell_tolerance must be > 0.")

        self.atom_property_names = validate_atom_property_names(self.atom_property_names)
        if not self.atom_property_table:
            self.atom_property_table = {key: tuple(value) for key, value in ATOM_PROPERTIES.items()}
        else:
            self.atom_property_table = {
                str(key): tuple(float(item) for item in value)
                for key, value in self.atom_property_table.items()
            }
        if self.dopant_elements is not None:
            self.dopant_elements = sort_dopant_elements(tuple(self.dopant_elements))

        shell_counts: dict[str, int] = {}
        for symbol, value in self.element_shell_counts.items():
            if symbol not in SUPPORTED_DOPANT_ELEMENTS:
                raise ValueError(
                    f"Unsupported local shell key {symbol!r}; expected one of {', '.join(SUPPORTED_DOPANT_ELEMENTS)}."
                )
            shell_counts[symbol] = int(value)
        for symbol, count in shell_counts.items():
            if count < 1:
                raise ValueError(f"local_shells.{symbol} must be >= 1.")
            if count > self.n_shells:
                raise ValueError(
                    f"local_shells.{symbol} = {count} is larger than n_shells = {self.n_shells}. "
                    "Increase n_shells or lower the element-specific shell count."
                )
        self.element_shell_counts = shell_counts

    def shell_count_for(self, symbol: str) -> int:
        return int(self.element_shell_counts.get(symbol, self.n_shells))


def load_structure(path: Path):
    return read_structure(path)


def species_symbols(structure) -> list[str]:
    return [str(site.specie.symbol) for site in structure]


def structure_dopant_elements(structure, source: Path | str) -> tuple[str, ...]:
    symbols = species_symbols(structure)
    allowed = {BASE_ELEMENT, *SUPPORTED_DOPANT_ELEMENTS}
    unsupported = sorted(set(symbols) - allowed)
    if unsupported:
        supported = ", ".join(SUPPORTED_DOPANT_ELEMENTS)
        raise ValueError(
            f"{source} contains unsupported elements: {', '.join(unsupported)}. "
            f"Allowed base element is {BASE_ELEMENT}; supported dopants are {supported}."
        )
    dopants = sort_dopant_elements(set(symbols) - {BASE_ELEMENT})
    if len(dopants) != 2:
        raise ValueError(
            f"{source} must contain exactly two different dopant elements besides {BASE_ELEMENT}; "
            f"found {len(dopants)} ({', '.join(dopants) or 'none'})."
        )
    return dopants


def validate_structure(
    structure,
    source: Path | str,
    allowed_dopants: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    dopants = structure_dopant_elements(structure, source)
    if allowed_dopants is not None:
        allowed_sorted = sort_dopant_elements(tuple(allowed_dopants))
        unknown = sorted(set(dopants) - set(allowed_sorted), key=atomic_number)
        if unknown:
            raise ValueError(
                f"{source} contains dopant elements not present in the training schema: "
                f"{', '.join(unknown)}. Training dopants: {', '.join(allowed_sorted)}."
            )
    return dopants


def validate_atom_property_names(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(str(name).strip() for name in names if str(name).strip())
    if not normalized:
        raise ValueError("atom_properties must contain at least one property name.")
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise ValueError(f"atom_properties contains duplicated names: {', '.join(duplicates)}.")
    unknown = sorted(set(normalized) - set(ATOM_PROPERTY_NAMES))
    if unknown:
        valid = ", ".join(ATOM_PROPERTY_NAMES)
        raise ValueError(f"Unknown atom_properties entries: {', '.join(unknown)}. Valid names: {valid}.")
    return normalized


def atom_property_indices(property_names: tuple[str, ...] | list[str]) -> list[int]:
    names = validate_atom_property_names(property_names)
    return [ATOM_PROPERTY_NAMES.index(name) for name in names]


def validate_atom_property_table(
    element_symbols: tuple[str, ...] | list[str],
    property_names: tuple[str, ...] | list[str],
    property_table: dict[str, tuple[float, ...] | list[float]] | None = None,
) -> None:
    table = property_table or ATOM_PROPERTIES
    names = validate_atom_property_names(property_names)
    indices = atom_property_indices(names)
    missing_elements = [symbol for symbol in element_symbols if symbol not in table]
    if missing_elements:
        raise ValueError(f"Missing atomic property rows for: {', '.join(missing_elements)}.")
    expected_width = len(ATOM_PROPERTY_NAMES)
    bad_width = [
        symbol for symbol in element_symbols
        if len(table[symbol]) != expected_width
    ]
    if bad_width:
        raise ValueError(
            "Atomic property table width mismatch for "
            f"{', '.join(bad_width)}; expected {expected_width} fields."
        )
    for symbol in element_symbols:
        values = np.asarray(table[symbol], dtype=np.float64)[indices]
        if not np.isfinite(values).all():
            bad_names = [name for name, value in zip(names, values) if not np.isfinite(value)]
            raise ValueError(f"Atomic properties for {symbol} contain non-finite values: {', '.join(bad_names)}.")


def raw_atom_properties(
    symbol: str,
    property_names: tuple[str, ...] | list[str] = ATOM_PROPERTY_NAMES,
    property_table: dict[str, tuple[float, ...] | list[float]] | None = None,
) -> np.ndarray:
    table = property_table or ATOM_PROPERTIES
    validate_atom_property_table((symbol,), property_names, table)
    values = np.asarray(table[symbol], dtype=np.float64)
    return values[atom_property_indices(property_names)]


def resolve_dopant_elements_from_records(records: list[SampleRecord]) -> tuple[str, ...]:
    dopants: set[str] = set()
    for record in records:
        structure = load_structure(record.poscar_path)
        dopants.update(validate_structure(structure, record.poscar_path))
    if not dopants:
        raise ValueError("Training split contains no valid dopant elements.")
    return sort_dopant_elements(dopants)


def resolved_element_shell_counts(options: FeatureOptions, dopant_elements: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in dopant_elements:
        count = int(options.element_shell_counts.get(symbol, options.n_shells))
        if count < 1:
            raise ValueError(f"local_shells.{symbol} must be >= 1.")
        if count > options.n_shells:
            raise ValueError(
                f"local_shells.{symbol} = {count} is larger than n_shells = {options.n_shells}. "
                "Increase n_shells or lower the element-specific shell count."
            )
        counts[symbol] = count
    return counts


def normalize_feature_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Complete and validate feature-schema metadata."""
    if "dopant_elements" in stats and "elements" in stats:
        dopants = sort_dopant_elements(tuple(stats["dopant_elements"]))
        elements = tuple(stats["elements"])
        expected_elements = element_symbols_for(dopants)
        if elements != expected_elements:
            raise ValueError(
                f"Stats element order {elements} does not match expected order {expected_elements}."
            )
        global_feature_mode = stats.get("global_feature_mode") or (
            "legacy_op" if is_legacy_op_dopant_set(dopants) else "dynamic"
        )
        declared_global_names = tuple(
            stats.get("global_feature_names")
            or stats.get("global", {}).get("names")
            or ()
        )
        expected_global_names = global_feature_names_for(dopants, global_feature_mode)
        if declared_global_names and declared_global_names != expected_global_names:
            raise ValueError(
                "Stats global feature names do not match the declared dopant schema; "
                "refusing to reorder features silently."
            )
        stats["dopant_elements"] = list(dopants)
        stats["elements"] = list(elements)
        stats["element_to_id"] = {symbol: index for index, symbol in enumerate(elements)}
        stats["global_feature_mode"] = global_feature_mode
        stats["global_feature_names"] = list(expected_global_names)
        stats.setdefault("feature_schema_version", "dynamic_dopant_schema_v1")
        stats.setdefault("num_dopant_elements", len(dopants))
        stats.setdefault("num_elements", len(elements))
        stats.setdefault("atom_property_names", list(stats.get("atom", {}).get("names", ATOM_PROPERTY_NAMES)))
        stats.setdefault("atom_property_table", {
            symbol: list(ATOM_PROPERTIES[symbol]) for symbol in elements
        })
        validate_atom_property_table(elements, stats["atom_property_names"], stats["atom_property_table"])
        if len(stats["atom"].get("mean", [])) != len(stats["atom_property_names"]):
            raise ValueError("Atom scaler dimension does not match atom_property_names.")
        if len(stats["global"].get("mean", [])) != len(stats["global_feature_names"]):
            raise ValueError("Global scaler dimension does not match global_feature_names.")
        stats.setdefault("input_dims", {
            "atom_property_dim": len(stats.get("atom_property_names") or stats["atom"].get("names", [])),
            "global_feature_dim": len(stats["global_feature_names"]),
            "n_shells": int(stats.get("n_shells", N_LOCAL_SHELLS)),
        })
        return stats

    elements = tuple(stats.get("elements", []))
    special_elements = tuple(stats.get("special_elements", []))
    global_names = tuple(stats.get("global_feature_names") or stats.get("global", {}).get("names", []))
    if elements == LEGACY_OP_ELEMENTS and special_elements == LEGACY_OP_DOPANT_ELEMENTS:
        if global_names and global_names != LEGACY_OP_GLOBAL_FEATURE_NAMES:
            raise ValueError("Legacy O/P stats have unexpected global feature names; refusing silent remap.")
        stats["dopant_elements"] = list(LEGACY_OP_DOPANT_ELEMENTS)
        stats["elements"] = list(LEGACY_OP_ELEMENTS)
        stats["element_to_id"] = element_mapping_for(LEGACY_OP_DOPANT_ELEMENTS)
        stats["special_elements"] = list(LEGACY_OP_DOPANT_ELEMENTS)
        stats["global_feature_mode"] = "legacy_op"
        stats["global_feature_names"] = list(LEGACY_OP_GLOBAL_FEATURE_NAMES)
        stats["feature_schema_version"] = "legacy_op_compat"
        stats["num_dopant_elements"] = len(LEGACY_OP_DOPANT_ELEMENTS)
        stats["num_elements"] = len(LEGACY_OP_ELEMENTS)
        stats.setdefault("atom_property_names", list(stats.get("atom", {}).get("names", ATOM_PROPERTY_NAMES)))
        stats.setdefault("atom_property_table", {
            symbol: list(ATOM_PROPERTIES[symbol]) for symbol in LEGACY_OP_ELEMENTS
        })
        validate_atom_property_table(LEGACY_OP_ELEMENTS, stats["atom_property_names"], stats["atom_property_table"])
        if len(stats["atom"].get("mean", [])) != len(stats["atom_property_names"]):
            raise ValueError("Legacy O/P atom scaler dimension does not match atom_property_names.")
        if len(stats["global"].get("mean", [])) != len(LEGACY_OP_GLOBAL_FEATURE_NAMES):
            raise ValueError("Legacy O/P global scaler dimension does not match the 20D feature order.")
        stats.setdefault("element_shell_counts", {
            symbol: int(stats.get("n_shells", N_LOCAL_SHELLS))
            for symbol in LEGACY_OP_DOPANT_ELEMENTS
        })
        stats.setdefault("shell_tolerance", SHELL_DISTANCE_TOLERANCE)
        stats.setdefault("input_dims", {
            "atom_property_dim": len(stats["atom_property_names"]),
            "global_feature_dim": len(LEGACY_OP_GLOBAL_FEATURE_NAMES),
            "n_shells": int(stats.get("n_shells", N_LOCAL_SHELLS)),
        })
        return stats

    raise ValueError(
        "Stats/checkpoint is missing dopant schema metadata and cannot be safely interpreted. "
        "Only confirmed legacy O/P stats can be upgraded automatically."
    )


def compute_global_features(
    structure,
    labels: dict[str, float],
    supercell: str | None,
    dopant_elements: tuple[str, ...] | list[str] = LEGACY_OP_DOPANT_ELEMENTS,
    mode: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    del labels
    dopants = sort_dopant_elements(tuple(dopant_elements))
    resolved_mode = mode or ("legacy_op" if is_legacy_op_dopant_set(dopants) else "dynamic")
    symbols = species_symbols(structure)
    layers = assign_bilayer_indices(structure)
    upper_atom_count = int((layers == 1).sum())
    lower_atom_count = int((layers == 0).sum())

    m, n = infer_supercell_dims(supercell, len(structure))
    lx = float(np.linalg.norm(structure.lattice.matrix[0]))
    ly = float(np.linalg.norm(structure.lattice.matrix[1]))
    z_values = np.asarray(structure.cart_coords[:, 2], dtype=np.float64)
    layer_distance = float(z_values[layers == 1].mean() - z_values[layers == 0].mean())
    total_atoms = float(len(structure))
    base_values = [m, n, lx, ly, layer_distance, total_atoms]

    per_element: dict[str, dict[str, float]] = {}
    count_special = 0
    for symbol in dopants:
        total_count = int(symbols.count(symbol))
        lower_count = sum(1 for idx, current in enumerate(symbols) if layers[idx] == 0 and current == symbol)
        upper_count = sum(1 for idx, current in enumerate(symbols) if layers[idx] == 1 and current == symbol)
        min_count, max_count = sorted((lower_count, upper_count))
        lower_conc = float(lower_count) / max(float(lower_atom_count), 1.0)
        upper_conc = float(upper_count) / max(float(upper_atom_count), 1.0)
        min_conc, max_conc = sorted((lower_conc, upper_conc))
        count_special += total_count
        per_element[symbol] = {
            "count": float(total_count),
            "total_concentration": float(total_count) / max(total_atoms, 1.0),
            "lower_count": float(lower_count),
            "upper_count": float(upper_count),
            "min_layer_count": float(min_count),
            "max_layer_count": float(max_count),
            "min_layer_concentration": float(min_conc),
            "max_layer_concentration": float(max_conc),
        }

    if resolved_mode == "legacy_op":
        if dopants != LEGACY_OP_DOPANT_ELEMENTS:
            raise ValueError("legacy_op global features are only valid for O/P dopants.")
        count_o = per_element["O"]["count"]
        count_p = per_element["P"]["count"]
        values = np.asarray(
            [
                *base_values,
                count_o,
                count_p,
                count_o / max(total_atoms, 1.0),
                count_p / max(total_atoms, 1.0),
                float(count_special) / max(total_atoms, 1.0),
                per_element["O"]["min_layer_count"],
                per_element["O"]["max_layer_count"],
                per_element["P"]["min_layer_count"],
                per_element["P"]["max_layer_count"],
                per_element["O"]["min_layer_concentration"],
                per_element["O"]["max_layer_concentration"],
                per_element["P"]["min_layer_concentration"],
                per_element["P"]["max_layer_concentration"],
                float(count_special),
            ],
            dtype=np.float64,
        )
    else:
        values = np.asarray(
            [
                *base_values,
                *[per_element[symbol]["count"] for symbol in dopants],
                *[per_element[symbol]["total_concentration"] for symbol in dopants],
                *[per_element[symbol]["min_layer_count"] for symbol in dopants],
                *[per_element[symbol]["max_layer_count"] for symbol in dopants],
                *[per_element[symbol]["min_layer_concentration"] for symbol in dopants],
                *[per_element[symbol]["max_layer_concentration"] for symbol in dopants],
                float(count_special),
                float(count_special) / max(total_atoms, 1.0),
            ],
            dtype=np.float64,
        )

    metadata = {
        "n_atoms": len(structure),
        "n_special": count_special,
        f"n_{BASE_ELEMENT}": symbols.count(BASE_ELEMENT),
        "upper_atoms": upper_atom_count,
        "lower_atoms": lower_atom_count,
        "layer_distance": layer_distance,
        "Lx": lx,
        "Ly": ly,
        "supercell_m": m,
        "supercell_n": n,
        "dopant_elements_in_schema": ",".join(dopants),
        "global_feature_mode": resolved_mode,
        "concentration_denominator": "total concentrations use total_atoms; layer concentrations use atoms in that layer",
    }
    for symbol in dopants:
        row = per_element[symbol]
        metadata[f"n_{symbol}"] = int(row["count"])
        metadata[f"lower_{symbol}"] = int(row["lower_count"])
        metadata[f"upper_{symbol}"] = int(row["upper_count"])
        metadata[f"{symbol}_min_layer_count"] = row["min_layer_count"]
        metadata[f"{symbol}_max_layer_count"] = row["max_layer_count"]
        metadata[f"{symbol}_concentration"] = row["total_concentration"]
        metadata[f"{symbol}_min_layer_concentration"] = row["min_layer_concentration"]
        metadata[f"{symbol}_max_layer_concentration"] = row["max_layer_concentration"]
    expected_names = global_feature_names_for(dopants, resolved_mode)
    if len(values) != len(expected_names):
        raise ValueError(
            f"Global feature length mismatch: generated {len(values)} values, "
            f"but schema declares {len(expected_names)} names."
        )
    return values, metadata


def fit_feature_stats(
    records: list[SampleRecord],
    label_names: list[str],
    options: FeatureOptions,
) -> dict[str, Any]:
    dopant_elements = (
        sort_dopant_elements(tuple(options.dopant_elements))
        if options.dopant_elements is not None
        else resolve_dopant_elements_from_records(records)
    )
    element_symbols = element_symbols_for(dopant_elements)
    element_to_id = element_mapping_for(dopant_elements)
    global_feature_mode = "legacy_op" if is_legacy_op_dopant_set(dopant_elements) else "dynamic"
    global_feature_names = global_feature_names_for(dopant_elements, global_feature_mode)
    element_shell_counts = resolved_element_shell_counts(options, dopant_elements)
    validate_atom_property_table(element_symbols, options.atom_property_names, options.atom_property_table)

    atom_values: list[np.ndarray] = []
    global_values: list[np.ndarray] = []
    target_values: list[list[float]] = []

    for record in records:
        structure = load_structure(record.poscar_path)
        validate_structure(structure, record.poscar_path, allowed_dopants=dopant_elements)
        for symbol in species_symbols(structure):
            atom_values.append(raw_atom_properties(symbol, options.atom_property_names, options.atom_property_table))

        global_raw, _ = compute_global_features(
            structure,
            record.labels,
            record.supercell,
            dopant_elements=dopant_elements,
            mode=global_feature_mode,
        )
        global_values.append(global_raw)
        target_values.append([float(record.labels[name]) for name in label_names])

    atom_scaler = StandardScaler.fit(np.asarray(atom_values), dim=len(options.atom_property_names))
    global_scaler = StandardScaler.fit(np.asarray(global_values), dim=len(global_feature_names))
    target_scaler = StandardScaler.fit(np.asarray(target_values), dim=len(label_names))

    return {
        "feature_version": FEATURE_VERSION,
        "feature_schema_version": "dynamic_dopant_schema_v1",
        "atom": atom_scaler.to_dict(options.atom_property_names),
        "atom_property_names": list(options.atom_property_names),
        "atom_property_table": {
            symbol: list(options.atom_property_table[symbol]) for symbol in element_symbols
        },
        "global": global_scaler.to_dict(global_feature_names),
        "target": target_scaler.to_dict(label_names),
        "target_names": label_names,
        "elements": list(element_symbols),
        "element_to_id": element_to_id,
        "dopant_elements": list(dopant_elements),
        "special_elements": list(dopant_elements),
        "num_elements": len(element_symbols),
        "num_dopant_elements": len(dopant_elements),
        "layer_relations": list(LAYER_RELATIONS),
        "n_shells": int(options.n_shells),
        "element_shell_counts": element_shell_counts,
        "shell_definition": "first_equal_distance_groups",
        "shell_tolerance": float(options.shell_tolerance),
        "image_range": int(options.image_range),
        "legacy_shell_ranges": [list(item) for item in SHELL_RANGES],
        "window_edge_order": list(WINDOW_EDGE_ORDER),
        "global_feature_mode": global_feature_mode,
        "global_feature_names": list(global_feature_names),
        "structure_feature_definition": {
            "dimension_formula": "6*K+8, except legacy O/P keeps the original 20-dimensional order",
            "K": len(dopant_elements),
            "total_concentration_denominator": "total_atoms",
            "layer_concentration_denominator": "atom count in the corresponding physical layer",
        },
        "input_dims": {
            "atom_property_dim": len(options.atom_property_names),
            "global_feature_dim": len(global_feature_names),
            "n_shells": int(options.n_shells),
            "num_elements": len(element_symbols),
        },
    }


def atom_scaled_lookup(stats: dict[str, Any]) -> dict[str, np.ndarray]:
    stats = normalize_feature_stats(stats)
    scaler = StandardScaler.from_dict(stats["atom"])
    property_names = tuple(stats.get("atom_property_names") or stats["atom"].get("names", ATOM_PROPERTY_NAMES))
    property_names = validate_atom_property_names(property_names)
    table = stats["atom_property_table"]
    return {
        symbol: scaler.transform(raw_atom_properties(symbol, property_names, table))
        for symbol in stats["elements"]
    }


def make_edge_payload(
    *,
    symbols: list[str],
    layers: np.ndarray,
    atom_props_scaled: dict[str, np.ndarray],
    element_to_id: dict[str, int],
    dopant_elements: tuple[str, ...],
    start_index: int,
    end_index: int,
    center_node_index: int | None,
    shell_index: int,
    is_center_edge: bool,
) -> dict[str, Any]:
    start_symbol = symbols[start_index]
    end_symbol = symbols[end_index]
    start_is_center = center_node_index is not None and start_index == center_node_index
    end_is_center = center_node_index is not None and end_index == center_node_index
    return {
        "atom_props": np.stack(
            [atom_props_scaled[start_symbol], atom_props_scaled[end_symbol]],
            axis=0,
        ).astype(np.float32),
        "element_ids": np.asarray([element_to_id[start_symbol], element_to_id[end_symbol]], dtype=np.int64),
        "center_flags": np.asarray([float(start_is_center), float(end_is_center)], dtype=np.float32),
        "dopant_flags": np.asarray(
            [float(start_symbol in dopant_elements), float(end_symbol in dopant_elements)],
            dtype=np.float32,
        ),
        "prop_diff": (atom_props_scaled[end_symbol] - atom_props_scaled[start_symbol]).astype(np.float32),
        "center_edge": np.asarray(float(is_center_edge), dtype=np.float32),
        "shell": int(shell_index),
        "start_layer": int(layers[start_index]),
        "end_layer": int(layers[end_index]),
    }


def stack_windows(
    window_payloads: list[list[dict[str, Any]]],
    special_count: int,
    atom_prop_dim: int,
) -> dict[str, np.ndarray]:
    max_windows = max([len(items) for items in window_payloads] + [0])
    window_edge_count = len(WINDOW_EDGE_ORDER)
    arrays = {
        "local_edge_atom_props": np.zeros(
            (special_count, max_windows, window_edge_count, 2, atom_prop_dim),
            dtype=np.float32,
        ),
        "local_edge_element_ids": np.zeros(
            (special_count, max_windows, window_edge_count, 2),
            dtype=np.int64,
        ),
        "local_edge_center_flags": np.zeros(
            (special_count, max_windows, window_edge_count, 2),
            dtype=np.float32,
        ),
        "local_edge_dopant_flags": np.zeros(
            (special_count, max_windows, window_edge_count, 2),
            dtype=np.float32,
        ),
        "local_edge_prop_diff": np.zeros(
            (special_count, max_windows, window_edge_count, atom_prop_dim),
            dtype=np.float32,
        ),
        "local_edge_center_mask": np.zeros(
            (special_count, max_windows, window_edge_count),
            dtype=np.float32,
        ),
        "window_shell": np.full((special_count, max_windows), -1, dtype=np.int64),
        "window_mask": np.zeros((special_count, max_windows), dtype=np.bool_),
    }
    for special_pos, windows in enumerate(window_payloads):
        for window_pos, window in enumerate(windows):
            arrays["window_shell"][special_pos, window_pos] = window[0]["shell"]
            arrays["window_mask"][special_pos, window_pos] = True
            for edge_pos, edge in enumerate(window):
                arrays["local_edge_atom_props"][special_pos, window_pos, edge_pos] = edge["atom_props"]
                arrays["local_edge_element_ids"][special_pos, window_pos, edge_pos] = edge["element_ids"]
                arrays["local_edge_center_flags"][special_pos, window_pos, edge_pos] = edge["center_flags"]
                arrays["local_edge_dopant_flags"][special_pos, window_pos, edge_pos] = edge["dopant_flags"]
                arrays["local_edge_prop_diff"][special_pos, window_pos, edge_pos] = edge["prop_diff"]
                arrays["local_edge_center_mask"][special_pos, window_pos, edge_pos] = edge["center_edge"]
    return arrays


def special_layer_relations(layers: np.ndarray, special_indices: list[int]) -> np.ndarray:
    count = len(special_indices)
    relations = np.zeros((count, count), dtype=np.int64)
    for i, atom_i in enumerate(special_indices):
        for j, atom_j in enumerate(special_indices):
            relations[i, j] = 0 if layers[atom_i] == layers[atom_j] else 1
    return relations


def build_descriptor(
    record: SampleRecord,
    stats: dict[str, Any],
    label_names: list[str],
    options: FeatureOptions,
) -> tuple[dict[str, torch.Tensor | dict[str, Any]], dict[str, Any]]:
    stats = normalize_feature_stats(stats)
    dopant_elements = tuple(stats["dopant_elements"])
    element_symbols = tuple(stats["elements"])
    element_to_id = {symbol: int(index) for symbol, index in stats["element_to_id"].items()}
    shell_count_by_element = {
        symbol: int(value)
        for symbol, value in dict(stats.get("element_shell_counts", {})).items()
    }
    global_feature_mode = stats.get("global_feature_mode")
    atom_property_names = tuple(stats.get("atom_property_names") or stats["atom"].get("names", ATOM_PROPERTY_NAMES))
    atom_prop_dim = len(atom_property_names)
    image_range = int(stats.get("image_range", options.image_range))
    shell_tolerance = float(stats.get("shell_tolerance", options.shell_tolerance))

    structure = load_structure(record.poscar_path)
    present_dopants = validate_structure(structure, record.poscar_path, allowed_dopants=dopant_elements)
    validate_atom_property_table(element_symbols, atom_property_names, stats["atom_property_table"])

    symbols = species_symbols(structure)
    layers = assign_bilayer_indices(structure)
    special_indices = [idx for idx, symbol in enumerate(symbols) if symbol in dopant_elements]
    atom_props_scaled = atom_scaled_lookup(stats)
    global_scaler = StandardScaler.from_dict(stats["global"])
    target_scaler = StandardScaler.from_dict(stats["target"])

    special_atom_props = []
    special_element_ids = []
    all_window_payloads: list[list[dict[str, Any]]] = []
    warnings: list[str] = []
    projection_cache: dict[tuple[int, int], int] = {}

    def projected(source_index: int, target_layer: int) -> int:
        key = (source_index, target_layer)
        if key not in projection_cache:
            projection_cache[key] = find_projection_atom(
                structure,
                layers,
                source_index,
                target_layer,
                image_range=image_range,
            )
        return projection_cache[key]

    def zero_edge_like(edge: dict[str, Any]) -> dict[str, Any]:
        return {
            key: np.zeros_like(value) if isinstance(value, np.ndarray) else value
            for key, value in edge.items()
        }

    def build_layer_center_windows(
        *,
        special_center_index: int,
        focal_center_index: int,
        opposite_center_index: int | None,
    ) -> list[list[dict[str, Any]]]:
        windows: list[list[dict[str, Any]]] = []
        shell_neighbors = find_equal_distance_shell_neighbors(
            structure,
            layers,
            focal_center_index,
            shell_count=shell_count_by_element.get(symbols[special_center_index], int(stats["n_shells"])),
            image_range=image_range,
            tolerance=shell_tolerance,
        )

        for shell_index, neighbors in enumerate(shell_neighbors):
            if not neighbors:
                warnings.append(
                    f"{record.sample_id}: special atom {special_center_index} "
                    f"has no equal-distance shell {shell_index} neighbors for center atom {focal_center_index}"
                )
                continue
            for position, current in enumerate(neighbors):
                left = neighbors[(position - 1) % len(neighbors)]
                right = neighbors[(position + 1) % len(neighbors)]
                same_layer_edges = (current, left, right)
                window: list[dict[str, Any]] = []

                for edge_offset, edge in enumerate(same_layer_edges):
                    window.append(
                        make_edge_payload(
                            symbols=symbols,
                            layers=layers,
                            atom_props_scaled=atom_props_scaled,
                            element_to_id=element_to_id,
                            dopant_elements=dopant_elements,
                            start_index=focal_center_index,
                            end_index=edge.atom_index,
                            center_node_index=focal_center_index,
                            shell_index=shell_index,
                            is_center_edge=(edge_offset == 0),
                        )
                    )

                for edge in same_layer_edges:
                    if opposite_center_index is None:
                        zero = zero_edge_like(window[0])
                        zero["center_edge"] = np.asarray(0.0, dtype=np.float32)
                        window.append(zero)
                        continue
                    try:
                        projected_neighbor = projected(edge.atom_index, int(layers[opposite_center_index]))
                        window.append(
                            make_edge_payload(
                                symbols=symbols,
                                layers=layers,
                                atom_props_scaled=atom_props_scaled,
                                element_to_id=element_to_id,
                                dopant_elements=dopant_elements,
                                start_index=opposite_center_index,
                                end_index=projected_neighbor,
                                center_node_index=None,
                                shell_index=shell_index,
                                is_center_edge=False,
                            )
                        )
                    except Exception as exc:
                        warnings.append(
                            f"{record.sample_id}: cannot project neighbor atom {edge.atom_index}: {exc}"
                        )
                        zero = zero_edge_like(window[0])
                        zero["center_edge"] = np.asarray(0.0, dtype=np.float32)
                        window.append(zero)

                windows.append(window)
        return windows

    for center_index in special_indices:
        center_symbol = symbols[center_index]
        center_layer = int(layers[center_index])
        opposite_layer = 1 - center_layer
        special_atom_props.append(atom_props_scaled[center_symbol].astype(np.float32))
        special_element_ids.append(element_to_id[center_symbol])

        center_windows: list[list[dict[str, Any]]] = []
        try:
            projected_center: int | None = projected(center_index, opposite_layer)
        except Exception as exc:
            projected_center = None
            warnings.append(f"{record.sample_id}: cannot project center atom {center_index}: {exc}")

        center_windows.extend(
            build_layer_center_windows(
                special_center_index=center_index,
                focal_center_index=center_index,
                opposite_center_index=projected_center,
            )
        )
        if projected_center is not None:
            center_windows.extend(
                build_layer_center_windows(
                    special_center_index=center_index,
                    focal_center_index=projected_center,
                    opposite_center_index=center_index,
                )
            )
        all_window_payloads.append(center_windows)

    local_arrays = stack_windows(all_window_payloads, len(special_indices), atom_prop_dim)
    special_distances = special_pair_distances(
        structure,
        special_indices,
        image_range=image_range,
    ).astype(np.float32)
    np.fill_diagonal(special_distances, 0.0)
    relation_matrix = special_layer_relations(layers, special_indices)

    global_raw, structure_metadata = compute_global_features(
        structure,
        record.labels,
        record.supercell,
        dopant_elements=dopant_elements,
        mode=global_feature_mode,
    )
    global_features = global_scaler.transform(global_raw)
    target_raw = np.asarray([record.labels[name] for name in label_names], dtype=np.float64)
    target_scaled = target_scaler.transform(target_raw)

    metadata = {
        **record.to_metadata(),
        **structure_metadata,
        "dopant_elements_in_structure": ",".join(present_dopants),
        "dopant_elements_in_model": ",".join(dopant_elements),
        "element_symbols_in_model": ",".join(element_symbols),
        "warnings": "; ".join(warnings),
    }

    sample = {
        **{key: torch.from_numpy(value) for key, value in local_arrays.items()},
        "special_atom_props": torch.from_numpy(np.stack(special_atom_props, axis=0).astype(np.float32)),
        "special_element_ids": torch.tensor(special_element_ids, dtype=torch.long),
        "special_mask": torch.ones(len(special_indices), dtype=torch.bool),
        "special_distances": torch.from_numpy(special_distances),
        "special_layer_relations": torch.from_numpy(relation_matrix),
        "global_features": torch.from_numpy(global_features.astype(np.float32)),
        "target": torch.from_numpy(target_scaled.astype(np.float32)),
        "target_raw": torch.from_numpy(target_raw.astype(np.float32)),
        "metadata": metadata,
    }
    return sample, metadata


def save_stats(stats: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_stats(path: Path) -> dict[str, Any]:
    return normalize_feature_stats(json.loads(path.read_text(encoding="utf-8")))
