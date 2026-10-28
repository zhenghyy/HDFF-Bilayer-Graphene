"""Domain constants for doped AA bilayer graphene descriptors."""

from __future__ import annotations

from dopant_graph.atom_features import load_atom_feature_table

BASE_ELEMENT = "C"
SUPPORTED_DOPANT_ELEMENTS = ("B", "N", "O", "F", "Si", "P", "S", "Ge", "As", "Se", "Sb")
LEGACY_OP_DOPANT_ELEMENTS = ("O", "P")
LEGACY_OP_ELEMENTS = (BASE_ELEMENT, *LEGACY_OP_DOPANT_ELEMENTS)

ELEMENTS = LEGACY_OP_ELEMENTS
SPECIAL_ELEMENTS = LEGACY_OP_DOPANT_ELEMENTS

FEATURE_VERSION = "hdff_net_config_atom_features_v2"
N_LOCAL_SHELLS = 5
SHELL_DISTANCE_TOLERANCE = 5.0e-2

SHELL_RANGES = (
    (1.4, 1.5),
    (2.4, 2.5),
    (2.8, 2.9),
    (3.7, 3.8),
    (4.2, 4.3),
)

WINDOW_EDGE_ORDER = (
    "center",
    "left",
    "right",
    "opposite_center",
    "opposite_left",
    "opposite_right",
)

ATOM_PROPERTY_NAMES, ATOM_PROPERTIES = load_atom_feature_table(
    required_elements=(BASE_ELEMENT, *SUPPORTED_DOPANT_ELEMENTS)
)

LEGACY_OP_GLOBAL_FEATURE_NAMES = (
    "supercell_m",
    "supercell_n",
    "Lx",
    "Ly",
    "layer_distance",
    "total_atoms",
    "num_O",
    "num_P",
    "O_concentration",
    "P_concentration",
    "special_concentration",
    "O_min_layer_count",
    "O_max_layer_count",
    "P_min_layer_count",
    "P_max_layer_count",
    "O_min_layer_concentration",
    "O_max_layer_concentration",
    "P_min_layer_concentration",
    "P_max_layer_concentration",
    "num_special",
)
GLOBAL_FEATURE_NAMES = LEGACY_OP_GLOBAL_FEATURE_NAMES

LAYER_RELATIONS = ("same", "cross")


def atomic_number(symbol: str) -> float:
    if symbol not in ATOM_PROPERTIES:
        raise ValueError(f"Missing atomic properties for element {symbol!r}.")
    try:
        index = ATOM_PROPERTY_NAMES.index("atomic_number")
    except ValueError as exc:
        raise ValueError("The config atom-feature schema has no 'Atomic number' field.") from exc
    return float(ATOM_PROPERTIES[symbol][index])


def sort_dopant_elements(elements: list[str] | tuple[str, ...] | set[str]) -> tuple[str, ...]:
    unknown = sorted(set(elements) - set(SUPPORTED_DOPANT_ELEMENTS))
    if unknown:
        supported = ", ".join(SUPPORTED_DOPANT_ELEMENTS)
        raise ValueError(f"Unsupported dopant elements: {', '.join(unknown)}. Supported dopants: {supported}.")
    return tuple(sorted(set(elements), key=atomic_number))


def element_symbols_for(dopant_elements: list[str] | tuple[str, ...] | set[str]) -> tuple[str, ...]:
    return (BASE_ELEMENT, *sort_dopant_elements(dopant_elements))


def element_mapping_for(dopant_elements: list[str] | tuple[str, ...] | set[str]) -> dict[str, int]:
    return {symbol: index for index, symbol in enumerate(element_symbols_for(dopant_elements))}


def is_legacy_op_dopant_set(dopant_elements: list[str] | tuple[str, ...] | set[str]) -> bool:
    return sort_dopant_elements(dopant_elements) == LEGACY_OP_DOPANT_ELEMENTS


def global_feature_names_for(
    dopant_elements: list[str] | tuple[str, ...] | set[str],
    mode: str | None = None,
) -> tuple[str, ...]:
    dopants = sort_dopant_elements(dopant_elements)
    resolved_mode = mode or ("legacy_op" if dopants == LEGACY_OP_DOPANT_ELEMENTS else "dynamic")
    if resolved_mode == "legacy_op":
        if dopants != LEGACY_OP_DOPANT_ELEMENTS:
            raise ValueError("legacy_op global features are only valid for O/P dopants.")
        return LEGACY_OP_GLOBAL_FEATURE_NAMES
    if resolved_mode != "dynamic":
        raise ValueError(f"Unknown global feature mode {resolved_mode!r}.")

    names: list[str] = [
        "supercell_m",
        "supercell_n",
        "Lx",
        "Ly",
        "layer_distance",
        "total_atoms",
    ]
    names.extend(f"num_{symbol}" for symbol in dopants)
    names.extend(f"{symbol}_concentration" for symbol in dopants)
    names.extend(f"{symbol}_min_layer_count" for symbol in dopants)
    names.extend(f"{symbol}_max_layer_count" for symbol in dopants)
    names.extend(f"{symbol}_min_layer_concentration" for symbol in dopants)
    names.extend(f"{symbol}_max_layer_concentration" for symbol in dopants)
    names.extend(["num_special", "special_concentration"])
    return tuple(names)


def element_one_hot(symbol: str, element_symbols: tuple[str, ...] = ELEMENTS) -> list[float]:
    if symbol not in element_symbols:
        raise ValueError(f"Unsupported element {symbol!r}; expected one of {element_symbols}.")
    return [1.0 if symbol == item else 0.0 for item in element_symbols]


def element_id(symbol: str, element_symbols: tuple[str, ...] = ELEMENTS) -> int:
    if symbol not in element_symbols:
        raise ValueError(f"Unsupported element {symbol!r}; expected one of {element_symbols}.")
    return element_symbols.index(symbol)


def special_type_one_hot(
    symbol: str,
    dopant_elements: tuple[str, ...] = SPECIAL_ELEMENTS,
) -> list[float]:
    if symbol not in dopant_elements:
        raise ValueError(f"Unsupported dopant element {symbol!r}; expected one of {dopant_elements}.")
    return [1.0 if symbol == item else 0.0 for item in dopant_elements]
