# FedHAT: Federated Heterogeneity-Aware Training for Cross-Country Iris Verification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Official repository for the paper by **Owais Ali Khan** and **Imtiaz Ahmed Taj**<br>
*Department of Electrical Engineering, Capital University of Science and Technology (CUST), Islamabad, Pakistan*

> **Status:** Under review at *Expert Systems with Applications*.

## Overview

A federated learning framework for non-IID, heterogeneous biometric datasets. FedHAT replaces sample-count aggregation with a **heterogeneity-aware** strategy that prevents large clients from dominating training and improves robustness on underrepresented datasets.

## Problem

Standard federated methods (e.g., FedAvg) weight clients by local sample count, implicitly assuming larger datasets produce more reliable updates. In cross-country iris verification this assumption breaks down:

- Clients differ in sensor, country, and acquisition conditions (non-IID).
- Identity counts and dataset sizes are highly imbalanced.
- Large clients dominate aggregation.

The result is biased global models and poor generalization to smaller, underrepresented clients.

## Approach

FedHAT scores each client's contribution from:

- Dataset size
- Identity richness
- Validation quality (EER / TAR)
- Rarity-aware weighting

It runs in two sequential stages:

1. **Rule-based warmup** — aggregation weights are set from dataset statistics and validation indicators.
2. **Learned aggregation** — clients are adaptively reweighted using a polynomial regression model fit on observed validation utility.

Backbone: SwinV2-Tiny Siamese network trained with metric-learning losses (batch-hard triplet + supervised contrastive).

## Repository Contents

### Preprocessing notebooks (run in order)

- `U-Net_Iris_Segmentation_Code.ipynb`
  Iris segmentation using Worldcoin's pretrained U-Net (ONNX format) → binary iris/pupil masks.
- `Iris_Isolation_Code-1.ipynb`
  Isolates the iris region from original images using masks; applies high-boost sharpening + random CLAHE/gamma enhancement.
- `Iris_Rubber_Sheet_Normalization.ipynb`
  Robust Daugman rubber-sheet normalization (64×512 polar format) with automatic circle fitting and cleanup.

### Training and evaluation

- `FedHAT.ipynb`
  Full FedHAT training pipeline (warmup heuristic → learned polynomial aggregation) using the SwinV2-Tiny Siamese network.
- `FL_Baselines.ipynb`
  Federated baselines used in the paper, under an identical backbone, loss, sampling, and schedule: **FedAvg, FedProx, FedYogi, FedAdam, MOON, FedNova, SCAFFOLD**. Local optimization uses AdamW for all methods except FedNova, which uses vanilla SGD (its normalization is defined only for solvers whose update is a linear combination of local gradients).
- `FL_ROC_AUC_Compute.ipynb`
  Loads a trained global model, computes ROC-AUC, EER, TAR@1% and TAR@0.1% FAR on identity-disjoint splits, and plots multi-panel ROC curves.

### JSON splits (provided)

- `china_split.json`, `czech_split.json`, `india_split.json`, `iran_split.json`, `iraq_split.json`, `malaysia_split.json`, `pakistan_split.json`
  Identity-disjoint train/val/test splits (per eye as identity).
  **Important:** these JSON files contain **hardcoded absolute paths** pointing to the author's local directories — see setup notes below.

### Trained models

The FedHAT global-model checkpoints for all three seeds (42, 123, 2025) are available on Google Drive:
[Download models here](https://drive.google.com/drive/folders/1_cC6EgJKle6MFKphfWbafrjZRC0YHI3J?usp=drive_link)

Seed **123** is the representative run (closest to the reported mean across all metrics); use it if you only need a single checkpoint.

### Result logs

Per-seed, per-client metric logs (CSV) for FedHAT and all baselines are provided in `results/`. These back the mean ± std reported in the paper and can be inspected without downloading any checkpoint.

## Reproducibility

- Training uses fixed seeds (42, 123, 2025) with deterministic settings enabled.
- To regenerate results: download the public datasets → run the preprocessing notebooks in order → correct paths in the notebooks and splits → run `FedHAT.ipynb` and `FL_Baselines.ipynb` → evaluate with `FL_ROC_AUC_Compute.ipynb`.
- Reported metrics are the mean ± std over the three seeds.

## Important Setup Notes (avoid errors!)

All notebooks currently use **hardcoded absolute paths** (e.g., `C:\Users\awais\OneDrive\Desktop\Thesis\...`). To run them on your machine:

1. Create your own folder structure for raw/preprocessed images.
2. Update all paths in the notebooks:
   - Input directories for original images
   - Output directories for masks, isolated iris, normalized images
   - Path to the downloaded model checkpoint
3. Update paths inside the JSON splits:
   - Open each `_split.json` file
   - Replace the old absolute paths with your new ones (or use relative paths if possible)
   - Save the modified JSONs

This is required because biometric datasets cannot be shared publicly, and the splits were created on the author's local setup.

## Datasets Used

| Dataset        | Country     | Type        | Source / Access                          |
|----------------|-------------|-------------|------------------------------------------|
| CUST-Iris      | Pakistan    | NIR         | Available via DOI (Mendeley Data: https://doi.org/10.17632/3j6skjpsng.2) |
| CASIA-Interval | China       | NIR         | Public (CASIA website)                   |
| UPOL           | Czech Rep.  | Visible     | Public                                   |
| IITD           | India       | NIR         | Public                                   |
| AMF            | Iraq        | NIR         | Public                                   |
| MMU V1         | Malaysia    | Visible     | Public                                   |
| UTIRIS         | Iran        | NIR/Visible | Public                                   |

Download the public datasets, apply the preprocessing notebooks in order (segmentation → isolation → normalization), then use the provided splits (after path correction).

## Requirements

```bash
pip install torch torchvision timm onnxruntime opencv-python tqdm scikit-learn matplotlib pillow numpy
```

Tested with Python 3.9–3.11 and PyTorch 2.0+.

## Citation

This work has been accepted for publication in Expert Systems with Applications.

```bibtex
@article{khan2026fedhat,
  title   = {FedHAT: Federated Heterogeneity-Aware Training for Cross-Country Iris Verification},
  author  = {Khan, Owais Ali and Taj, Imtiaz Ahmed},
  year    = {2026},
  journal = {Expert Systems with Applications},
}
```

## License

This project is licensed under the MIT License — see `LICENSE` for details.

## Contact

**Owais Ali Khan** — awais.ali.khan610@gmail.com
**Imtiaz Ahmed Taj** — iataj777@gmail.com

For access to the CUST-Iris dataset or any questions, feel free to reach out.
