"""Torch dataset and padding collate for HDFF-Net descriptors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class DopantGraphDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, split: str):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.path = self.dataset_dir / "processed" / f"{split}.pt"
        if not self.path.is_file():
            raise FileNotFoundError(f"Missing processed split: {self.path}")
        self.samples = torch.load(self.path, map_location="cpu")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def _tensor_dim(sample: dict[str, Any], key: str) -> int:
    return int(sample[key].shape[-1])


def collate_dopant_graph(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    batch_size = len(samples)
    max_special = max(int(sample["special_atom_props"].shape[0]) for sample in samples)
    max_windows = max(max(1, int(sample["local_edge_atom_props"].shape[1])) for sample in samples)
    window_edges = int(samples[0]["local_edge_atom_props"].shape[2])
    atom_prop_dim = int(samples[0]["local_edge_atom_props"].shape[-1])
    global_dim = _tensor_dim(samples[0], "global_features")
    target_dim = _tensor_dim(samples[0], "target")

    local_edge_atom_props = torch.zeros(
        batch_size, max_special, max_windows, window_edges, 2, atom_prop_dim, dtype=torch.float32
    )
    local_edge_element_ids = torch.zeros(
        batch_size, max_special, max_windows, window_edges, 2, dtype=torch.long
    )
    local_edge_center_flags = torch.zeros(
        batch_size, max_special, max_windows, window_edges, 2, dtype=torch.float32
    )
    local_edge_dopant_flags = torch.zeros(
        batch_size, max_special, max_windows, window_edges, 2, dtype=torch.float32
    )
    local_edge_prop_diff = torch.zeros(
        batch_size, max_special, max_windows, window_edges, atom_prop_dim, dtype=torch.float32
    )
    local_edge_center_mask = torch.zeros(batch_size, max_special, max_windows, window_edges, dtype=torch.float32)

    window_shell = torch.full((batch_size, max_special, max_windows), -1, dtype=torch.long)
    window_mask = torch.zeros(batch_size, max_special, max_windows, dtype=torch.bool)
    special_atom_props = torch.zeros(batch_size, max_special, atom_prop_dim, dtype=torch.float32)
    special_element_ids = torch.zeros(batch_size, max_special, dtype=torch.long)
    special_mask = torch.zeros(batch_size, max_special, dtype=torch.bool)
    special_distances = torch.zeros(batch_size, max_special, max_special, dtype=torch.float32)
    special_layer_relations = torch.zeros(batch_size, max_special, max_special, dtype=torch.long)
    global_features = torch.zeros(batch_size, global_dim, dtype=torch.float32)
    target = torch.zeros(batch_size, target_dim, dtype=torch.float32)
    target_raw = torch.zeros(batch_size, target_dim, dtype=torch.float32)
    metadata = []

    for batch_index, sample in enumerate(samples):
        special_count = int(sample["special_atom_props"].shape[0])
        window_count = int(sample["local_edge_atom_props"].shape[1])
        if special_count == 0:
            raise ValueError("A sample with zero special points reached the collate function.")

        special_atom_props[batch_index, :special_count] = sample["special_atom_props"]
        special_element_ids[batch_index, :special_count] = sample["special_element_ids"]
        special_mask[batch_index, :special_count] = sample["special_mask"]
        special_distances[batch_index, :special_count, :special_count] = sample["special_distances"]
        special_layer_relations[batch_index, :special_count, :special_count] = sample["special_layer_relations"]
        global_features[batch_index] = sample["global_features"]
        target[batch_index] = sample["target"]
        target_raw[batch_index] = sample["target_raw"]
        metadata.append(sample["metadata"])

        if window_count > 0:
            dst = (batch_index, slice(0, special_count), slice(0, window_count))
            local_edge_atom_props[dst] = sample["local_edge_atom_props"]
            local_edge_element_ids[dst] = sample["local_edge_element_ids"]
            local_edge_center_flags[dst] = sample["local_edge_center_flags"]
            local_edge_dopant_flags[dst] = sample["local_edge_dopant_flags"]
            local_edge_prop_diff[dst] = sample["local_edge_prop_diff"]
            local_edge_center_mask[dst] = sample["local_edge_center_mask"]
            window_shell[dst] = sample["window_shell"]
            window_mask[dst] = sample["window_mask"]

    return {
        "local_edge_atom_props": local_edge_atom_props,
        "local_edge_element_ids": local_edge_element_ids,
        "local_edge_center_flags": local_edge_center_flags,
        "local_edge_dopant_flags": local_edge_dopant_flags,
        "local_edge_prop_diff": local_edge_prop_diff,
        "local_edge_center_mask": local_edge_center_mask,
        "window_shell": window_shell,
        "window_mask": window_mask,
        "special_atom_props": special_atom_props,
        "special_element_ids": special_element_ids,
        "special_mask": special_mask,
        "special_distances": special_distances,
        "special_layer_relations": special_layer_relations,
        "global_features": global_features,
        "target": target,
        "target_raw": target_raw,
        "metadata": metadata,
    }


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
