"""Geometry helpers with two-dimensional periodic boundary conditions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ShellNeighbor:
    atom_index: int
    image: tuple[int, int]
    shell_index: int
    distance: float
    angle: float
    delta_frac_2d: np.ndarray
    delta_cart: np.ndarray


def lattice_ab(structure: Any) -> np.ndarray:
    return np.asarray(structure.lattice.matrix[:2], dtype=np.float64)


def frac_2d_displacement(
    frac_from: np.ndarray,
    frac_to: np.ndarray,
    image: tuple[int, int] = (0, 0),
) -> np.ndarray:
    return np.asarray(
        [
            frac_to[0] + image[0] - frac_from[0],
            frac_to[1] + image[1] - frac_from[1],
        ],
        dtype=np.float64,
    )


def frac2d_to_cart(delta_frac_2d: np.ndarray, ab_vectors: np.ndarray) -> np.ndarray:
    return delta_frac_2d[0] * ab_vectors[0] + delta_frac_2d[1] * ab_vectors[1]


def iter_images(image_range: int) -> Iterable[tuple[int, int]]:
    for tx in range(-image_range, image_range + 1):
        for ty in range(-image_range, image_range + 1):
            yield tx, ty


def minimum_image_2d(
    frac_from: np.ndarray,
    frac_to: np.ndarray,
    ab_vectors: np.ndarray,
    image_range: int = 1,
) -> tuple[float, tuple[int, int], np.ndarray, np.ndarray]:
    best: tuple[float, tuple[int, int], np.ndarray, np.ndarray] | None = None
    for image in iter_images(image_range):
        delta_frac = frac_2d_displacement(frac_from, frac_to, image)
        delta_cart = frac2d_to_cart(delta_frac, ab_vectors)
        distance = float(np.linalg.norm(delta_cart))
        if best is None or distance < best[0]:
            best = (distance, image, delta_frac, delta_cart)
    if best is None:
        raise RuntimeError("No periodic image was evaluated.")
    return best


def assign_bilayer_indices(structure: Any) -> np.ndarray:
    """Assign lower layer to 0 and upper layer to 1 from Cartesian z positions."""
    z_values = np.asarray(structure.cart_coords[:, 2], dtype=np.float64)
    if len(z_values) < 2:
        raise ValueError("A bilayer structure needs at least two atoms.")

    order = np.argsort(z_values)
    sorted_z = z_values[order]
    gaps = np.diff(sorted_z)
    if len(gaps) == 0:
        raise ValueError("Cannot split layers from a single z coordinate.")

    gap_index = int(np.argmax(gaps))
    threshold = 0.5 * (sorted_z[gap_index] + sorted_z[gap_index + 1])
    layers = (z_values > threshold).astype(np.int64)

    if int((layers == 0).sum()) == 0 or int((layers == 1).sum()) == 0:
        median = float(np.median(z_values))
        layers = (z_values > median).astype(np.int64)
    if int((layers == 0).sum()) == 0 or int((layers == 1).sum()) == 0:
        raise ValueError("Could not split the structure into two z layers.")
    return layers


def direction_angle(delta_cart: np.ndarray) -> float:
    angle = math.atan2(float(delta_cart[1]), float(delta_cart[0]))
    if angle < 0.0:
        angle += 2.0 * math.pi
    return angle


def find_shell_neighbors(
    structure: Any,
    layers: np.ndarray,
    center_index: int,
    shell_ranges: tuple[tuple[float, float], ...],
    image_range: int = 1,
) -> list[list[ShellNeighbor]]:
    frac = np.asarray(structure.frac_coords, dtype=np.float64)
    ab = lattice_ab(structure)
    center_layer = int(layers[center_index])
    shell_neighbors: list[list[ShellNeighbor]] = [[] for _ in shell_ranges]

    for atom_index in np.where(layers == center_layer)[0]:
        for image in iter_images(image_range):
            if atom_index == center_index and image == (0, 0):
                continue
            delta_frac = frac_2d_displacement(frac[center_index], frac[atom_index], image)
            delta_cart = frac2d_to_cart(delta_frac, ab)
            distance = float(np.linalg.norm(delta_cart))
            for shell_index, (lower, upper) in enumerate(shell_ranges):
                if lower <= distance <= upper:
                    shell_neighbors[shell_index].append(
                        ShellNeighbor(
                            atom_index=int(atom_index),
                            image=image,
                            shell_index=shell_index,
                            distance=distance,
                            angle=direction_angle(delta_cart),
                            delta_frac_2d=delta_frac,
                            delta_cart=delta_cart,
                        )
                    )
                    break

    for neighbors in shell_neighbors:
        neighbors.sort(key=lambda item: (item.angle, item.distance, item.atom_index, item.image))
    return shell_neighbors


def find_equal_distance_shell_neighbors(
    structure: Any,
    layers: np.ndarray,
    center_index: int,
    shell_count: int,
    image_range: int = 1,
    tolerance: float = 1.0e-3,
) -> list[list[ShellNeighbor]]:
    """Group same-layer neighbors into the first equal-distance shells.

    Each physical atom contributes its nearest two-dimensional periodic image.
    Distances that differ by at most ``tolerance`` Angstrom are treated as the
    same discrete shell.
    """
    frac = np.asarray(structure.frac_coords, dtype=np.float64)
    ab = lattice_ab(structure)
    center_layer = int(layers[center_index])
    candidates: list[ShellNeighbor] = []

    for atom_index in np.where(layers == center_layer)[0]:
        if int(atom_index) == center_index:
            continue
        distance, image, delta_frac, delta_cart = minimum_image_2d(
            frac[center_index],
            frac[int(atom_index)],
            ab,
            image_range=image_range,
        )
        if distance <= tolerance:
            continue
        candidates.append(
            ShellNeighbor(
                atom_index=int(atom_index),
                image=image,
                shell_index=-1,
                distance=distance,
                angle=direction_angle(delta_cart),
                delta_frac_2d=delta_frac,
                delta_cart=delta_cart,
            )
        )

    candidates.sort(key=lambda item: (item.distance, item.angle, item.atom_index, item.image))
    shell_groups: list[list[ShellNeighbor]] = []
    shell_distances: list[float] = []
    for candidate in candidates:
        target_index = None
        for index, shell_distance in enumerate(shell_distances):
            if abs(candidate.distance - shell_distance) <= tolerance:
                target_index = index
                break
        if target_index is None:
            if len(shell_groups) >= shell_count:
                break
            target_index = len(shell_groups)
            shell_groups.append([])
            shell_distances.append(candidate.distance)
        shell_groups[target_index].append(
            ShellNeighbor(
                atom_index=candidate.atom_index,
                image=candidate.image,
                shell_index=target_index,
                distance=candidate.distance,
                angle=candidate.angle,
                delta_frac_2d=candidate.delta_frac_2d,
                delta_cart=candidate.delta_cart,
            )
        )

    while len(shell_groups) < shell_count:
        shell_groups.append([])
    for neighbors in shell_groups:
        neighbors.sort(key=lambda item: (item.angle, item.distance, item.atom_index, item.image))
    return shell_groups


def find_projection_atom(
    structure: Any,
    layers: np.ndarray,
    source_index: int,
    target_layer: int,
    image_range: int = 1,
) -> int:
    frac = np.asarray(structure.frac_coords, dtype=np.float64)
    ab = lattice_ab(structure)
    candidates = np.where(layers == target_layer)[0]
    if len(candidates) == 0:
        raise ValueError(f"No atoms found in target layer {target_layer}.")

    best_index = int(candidates[0])
    best_distance = float("inf")
    for atom_index in candidates:
        distance, _, _, _ = minimum_image_2d(
            frac[source_index],
            frac[int(atom_index)],
            ab,
            image_range=image_range,
        )
        if distance < best_distance:
            best_distance = distance
            best_index = int(atom_index)
    return best_index


def special_pair_distances(
    structure: Any,
    special_indices: list[int],
    image_range: int = 1,
) -> np.ndarray:
    frac = np.asarray(structure.frac_coords, dtype=np.float64)
    ab = lattice_ab(structure)
    count = len(special_indices)
    distances = np.zeros((count, count), dtype=np.float32)
    for i, atom_i in enumerate(special_indices):
        for j, atom_j in enumerate(special_indices):
            if i == j:
                continue
            distance, _, _, _ = minimum_image_2d(
                frac[atom_i],
                frac[atom_j],
                ab,
                image_range=image_range,
            )
            distances[i, j] = distance
    return distances


def infer_supercell_dims(supercell: str | None, atom_count: int) -> tuple[float, float]:
    if supercell:
        match = re.search(r"(\d+)\s*x\s*(\d+)", supercell.lower())
        if match:
            return float(match.group(1)), float(match.group(2))

    primitive_cells = atom_count / 4.0
    root = round(math.sqrt(primitive_cells))
    if root > 0 and abs(root * root - primitive_cells) < 1e-6:
        return float(root), float(root)
    return 0.0, 0.0


def infer_supercell_from_path(parts: Iterable[str]) -> str | None:
    for part in reversed(list(parts)):
        match = re.search(r"(?:sc[_-]?)?(\d+\s*x\s*\d+)", part.lower())
        if match:
            return "sc_" + match.group(1).replace(" ", "")
    return None
