"""Structure loading with a lightweight POSCAR fallback when pymatgen is absent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    from pymatgen.core import Structure as PymatgenStructure
except ModuleNotFoundError:  # pragma: no cover - exercised only without pymatgen
    PymatgenStructure = None


@dataclass(frozen=True)
class SimpleLattice:
    matrix: np.ndarray


@dataclass(frozen=True)
class SimpleSpecie:
    symbol: str


@dataclass(frozen=True)
class SimpleSite:
    specie: SimpleSpecie


class SimpleStructure:
    def __init__(self, lattice_matrix: np.ndarray, symbols: list[str], frac_coords: np.ndarray):
        self.lattice = SimpleLattice(np.asarray(lattice_matrix, dtype=np.float64))
        self._symbols = list(symbols)
        self.frac_coords = np.asarray(frac_coords, dtype=np.float64)
        self.cart_coords = self.frac_coords @ self.lattice.matrix

    def __len__(self) -> int:
        return len(self._symbols)

    def __iter__(self) -> Iterator[SimpleSite]:
        for symbol in self._symbols:
            yield SimpleSite(SimpleSpecie(symbol))


def _parse_float_triplet(line: str) -> list[float]:
    parts = line.split()
    if len(parts) < 3:
        raise ValueError(f"Expected three coordinate values, got: {line!r}")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def read_poscar_light(path: Path) -> SimpleStructure:
    raw_lines = [line.rstrip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    lines = [line.strip() for line in raw_lines if line.strip()]
    if len(lines) < 8:
        raise ValueError(f"POSCAR is too short: {path}")

    scale = float(lines[1].split()[0])
    lattice = np.asarray([_parse_float_triplet(lines[index]) for index in range(2, 5)], dtype=np.float64)
    if scale < 0:
        volume = abs(float(np.linalg.det(lattice)))
        scale_factor = ((-scale) / volume) ** (1.0 / 3.0)
    else:
        scale_factor = scale
    lattice = lattice * scale_factor

    element_tokens = lines[5].split()
    if all(token.replace(".", "", 1).isdigit() for token in element_tokens):
        raise ValueError("Fallback POSCAR reader requires VASP5 element symbols on line 6.")
    count_tokens = lines[6].split()
    counts = [int(float(token)) for token in count_tokens]
    if len(element_tokens) != len(counts):
        raise ValueError("Element and count lines have different lengths.")

    mode_index = 7
    if lines[mode_index].lower().startswith("s"):
        mode_index += 1
    coord_mode = lines[mode_index].lower()
    direct = coord_mode.startswith("d")
    cartesian = coord_mode.startswith(("c", "k"))
    if not direct and not cartesian:
        raise ValueError(f"Unknown POSCAR coordinate mode: {lines[mode_index]!r}")

    total_atoms = sum(counts)
    coord_start = mode_index + 1
    coord_lines = lines[coord_start:coord_start + total_atoms]
    if len(coord_lines) != total_atoms:
        raise ValueError(f"POSCAR has {len(coord_lines)} coordinates, expected {total_atoms}.")
    coords = np.asarray([_parse_float_triplet(line) for line in coord_lines], dtype=np.float64)
    if direct:
        frac_coords = coords
    else:
        cart_coords = coords * scale_factor
        frac_coords = cart_coords @ np.linalg.inv(lattice)
    frac_coords = frac_coords % 1.0

    symbols: list[str] = []
    for symbol, count in zip(element_tokens, counts):
        symbols.extend([symbol] * count)
    return SimpleStructure(lattice, symbols, frac_coords)


def read_structure(path: Path):
    if PymatgenStructure is not None:
        return PymatgenStructure.from_file(str(path))
    return read_poscar_light(path)
