# HDFF: Four-Property Prediction for Bilayer Graphene

This project provides pretrained HDFF models for predicting four relaxed-state properties of doped bilayer graphene directly from initial, unrelaxed structures:

- Formation energy
- Mean interlayer spacing
- Fermi level
- Total electronic energy

The framework integrates local atomic environments, interactions between dopant sites, and structure-level information. Only initial structures are used as model inputs; relaxed atomic coordinates and DFTB+ calculation outputs are not required for prediction.

## Files

- `data/initial_structures/`: 650 initial structures in `.vasp` format.
- `data/labels.csv`: DFTB+ reference values for the four properties.
- `models/`: Pretrained model checkpoints (`best.pt`).
- `dopant_graph/`: Atomic descriptor and graph construction modules.
- `hdff_model_portable/`: HDFF model architecture and inference modules.
- `MODEL_ARCHITECTURE.md`: Overview of the model architecture.
- `predict.py`: Script for property prediction and evaluation.
- `verify_package.py`: Script for checking package integrity and results.
- `results/`: Prediction outputs and evaluation metrics.

## Requirements and Usage

Python 3.10 or newer is required.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run property prediction:

```bash
python predict.py --device cpu --batch-size 8
```

Verify the results:

```bash
python verify_package.py
```

GPU inference is also supported by replacing `--device cpu` with `--device cuda`.

## Prediction Results

The prediction script generates:

- `results/predictions_650.csv`: Predicted and reference values for the four properties.
- `results/metrics_650.csv`: Evaluation metrics for the 650 structures.

The package is intended to reproduce predictions using the supplied pretrained models. The 650-structure collection includes some training samples, so its aggregate metrics should not be interpreted as independent test-set performance.