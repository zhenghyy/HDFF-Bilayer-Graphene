"""Check package completeness, hashes, and metric arithmetic without training code."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from predict import ROOT, TARGETS, load_labels, metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    rows = load_labels(ROOT / "data/labels.csv")
    ids = {row["structure_id"] for row in rows}
    structures = ROOT / "data/initial_structures"
    files = list(structures.glob("*.vasp"))
    assert len(rows) == manifest["structure_count"] == 650
    assert {path.stem for path in files} == ids
    assert len(files) == 650
    assert all("/" not in sid and "\\" not in sid and ".." not in sid for sid in ids)
    for target in TARGETS:
        checkpoint = ROOT / "models" / target / "best.pt"
        assert checkpoint.is_file()
        assert sha256(checkpoint) == manifest["models"][target]["checkpoint_sha256"]

    pred_path = ROOT / "results/predictions_650.csv"
    metric_path = ROOT / "results/metrics_650.csv"
    if pred_path.is_file() and metric_path.is_file():
        with pred_path.open(newline="", encoding="utf-8-sig") as handle:
            preds = list(csv.DictReader(handle))
        with metric_path.open(newline="", encoding="utf-8-sig") as handle:
            reported = {row["target"]: row for row in csv.DictReader(handle)}
        assert len(preds) == 650 and {row["structure_id"] for row in preds} == ids
        for target in TARGETS:
            true = [float(row[f"{target}_true"]) for row in preds]
            pred = [float(row[f"{target}_pred"]) for row in preds]
            assert all(math.isfinite(value) for value in true + pred)
            residual = [abs(p - t) for p, t in zip(pred, true)]
            assert all(abs(value - float(row[f"{target}_abs_error"])) < 1e-6
                       for value, row in zip(residual, preds))
            import numpy as np
            calculated = metrics(np.asarray(true), np.asarray(pred))
            assert int(reported[target]["n"]) == 650
            for name, value in calculated.items():
                assert abs(float(reported[target][name]) - value) < 1e-7
    print("Package verified: 650 unique structures, 4 checkpoints, metrics consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
