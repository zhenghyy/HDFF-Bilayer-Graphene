"""Predict four properties from initial POSCAR structures using bundled HDFF checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from dopant_graph.dataset import batch_to_device, collate_dopant_graph
from dopant_graph.featurizer import FeatureOptions, build_descriptor
from dopant_graph.io import SampleRecord
from hdff_model_portable.training_workspace.models.hdff_embedding_ablation_v1 import build_model


ROOT = Path(__file__).resolve().parent
TARGETS = (
    "formation_energy_eV",
    "interlayer_spacing_mean_A",
    "fermi_level_eV",
    "total_electronic_energy_eV",
)


def load_labels(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected = {"structure_id", *TARGETS}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(f"labels CSV must have exactly these columns: {sorted(expected)}")
        rows = list(reader)
    ids = [row["structure_id"].strip() for row in rows]
    if len(ids) != len(set(ids)) or any(not sid for sid in ids):
        raise ValueError("labels CSV contains duplicate or blank structure_id")
    if not rows:
        raise ValueError("labels CSV is empty")
    for row in rows:
        for target in TARGETS:
            value = float(row[target])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {target} for {row['structure_id']}")
    return rows


def options_from_stats(stats: dict) -> FeatureOptions:
    return FeatureOptions(
        n_shells=int(stats["n_shells"]),
        shell_tolerance=float(stats.get("shell_tolerance", 0.05)),
        image_range=int(stats.get("image_range", 1)),
        atom_property_names=tuple(stats["atom_property_names"]),
        element_shell_counts=dict(stats.get("element_shell_counts", {})),
        dopant_elements=tuple(stats["dopant_elements"]),
        atom_property_table=dict(stats["atom_property_table"]),
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state", "model_config", "feature_stats"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Invalid inference checkpoint: {path}")
    model = build_model(payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model = model.to(device).eval()
    return model, payload["feature_stats"]


def predict_one_target(
    label_rows: list[dict[str, str]],
    structures_dir: Path,
    target: str,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model, stats = load_checkpoint(checkpoint_path, device)
    if stats.get("target_names") != [target]:
        raise ValueError(f"Checkpoint {checkpoint_path} targets {stats.get('target_names')}, not {target}")
    options = options_from_stats(stats)
    mean, std = (float(stats["target"][key][0]) for key in ("mean", "std"))
    predictions: list[float] = []
    for offset in range(0, len(label_rows), batch_size):
        samples = []
        for row in label_rows[offset : offset + batch_size]:
            sid = row["structure_id"].strip()
            poscar = structures_dir / f"{sid}.vasp"
            if not poscar.is_file():
                raise FileNotFoundError(poscar)
            record = SampleRecord(
                sample_id=sid,
                poscar_path=poscar,
                labels={target: float(row[target])},
                source_dir=poscar.parent,
            )
            sample, _ = build_descriptor(record, stats, [target], options)
            samples.append(sample)
        batch = batch_to_device(collate_dopant_graph(samples), device)
        with torch.inference_mode():
            raw = model(batch).detach().cpu().numpy().reshape(-1)
        predictions.extend((raw * std + mean).astype(np.float64).tolist())
        print(f"{target}: {min(offset + len(samples), len(label_rows))}/{len(label_rows)}", flush=True)
    return np.asarray(predictions, dtype=np.float64)


def metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - true
    sst = float(np.square(true - true.mean()).sum())
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": float(1.0 - np.square(residual).sum() / sst) if sst > 0 else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "labels.csv")
    parser.add_argument("--structures", type=Path, default=ROOT / "data" / "initial_structures")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    rows = load_labels(args.labels)
    predictions = {}
    for target in TARGETS:
        checkpoint_path = args.models / target / "best.pt"
        predictions[target] = predict_one_target(
            rows, args.structures, target, checkpoint_path, device, args.batch_size
        )

    args.output.mkdir(parents=True, exist_ok=True)
    pred_path = args.output / "predictions_650.csv"
    fields = ["structure_id"]
    for target in TARGETS:
        fields.extend((f"{target}_true", f"{target}_pred", f"{target}_abs_error"))
    with pred_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            item: dict[str, str | float] = {"structure_id": row["structure_id"]}
            for target in TARGETS:
                true = float(row[target])
                pred = float(predictions[target][index])
                item[f"{target}_true"] = true
                item[f"{target}_pred"] = pred
                item[f"{target}_abs_error"] = abs(pred - true)
            writer.writerow(item)

    metric_path = args.output / "metrics_650.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("target", "unit", "n", "mae", "rmse", "r2"))
        writer.writeheader()
        for target in TARGETS:
            true = np.asarray([float(row[target]) for row in rows], dtype=np.float64)
            writer.writerow({
                "target": target,
                "unit": "A" if target == "interlayer_spacing_mean_A" else "eV",
                "n": len(rows),
                **metrics(true, predictions[target]),
            })
    print(f"Saved {pred_path} and {metric_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
