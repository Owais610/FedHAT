# FedHAT: Federated Heterogeneity-Aware Training for Cross-Country Iris Verification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Official repository for the paper:  
**FedHAT: Federated Heterogeneity-Aware Training for Cross-Country Iris Verification**  
Owais Ali Khan¹, Imtiaz Ahmed Taj²  
¹² Department of Electrical Engineering, Capital University of Science and Technology (CUST), Islamabad, Pakistan  

This repository provides the **complete end-to-end pipeline** (Jupyter notebooks) for reproducing the results in the paper, including preprocessing, normalization, federated training with the novel FedHAT aggregation, and final evaluation.

## Repository Contents

### Jupyter Notebooks
- `U-Net_Iris_Segmentation_Code.ipynb`  
  Iris segmentation using Worldcoin's pretrained U-Net (ONNX format) → binary iris/pupil masks.

- `Iris_Isolation_Code-1.ipynb`  
  Isolates iris region from original images using masks, applies high-boost sharpening + random CLAHE/gamma enhancement.

- `Iris_Rubber_Sheet_Normalization.ipynb`  
  Robust Daugman rubber-sheet normalization (64×512 polar format) with automatic circle fitting and cleanup.

- `FedHAT.ipynb`  
  Full federated learning training pipeline implementing **FedHAT** (warmup heuristic → learned polynomial aggregation) using SwinV2-Tiny Siamese network.

- `FL_ROC_AUC_Compute.ipynb`  
  Loads the trained global model, computes test metrics (ROC-AUC, EER, TAR@1% and 0.1% FAR) on identity-disjoint splits, and plots multi-panel ROC curves.

### JSON Splits (provided)
- `china_split.json`, `czech_split.json`, `india_split.json`, `iran_split.json`, `iraq_split.json`, `malaysia_split.json`, `pakistan_split.json`  
  Identity-disjoint train/val/test splits (per eye as identity).  
  **Important**: These JSON files contain **hardcoded absolute paths** pointing to the author's local directories.

### Trained Model
The best FedHAT global model checkpoint (`FHIR_POLY_EER_best.pt`) is available on Google Drive:  
[Download Model Here](https://drive.google.com/file/d/1cXg4xVDUHR7HMc17WJtrHXzR5j6lQ30n/view?usp=drive_link)  


## Important Setup Notes (Avoid Errors!)

All notebooks currently use **hardcoded absolute paths** (e.g., `C:\Users\awais\OneDrive\Desktop\Thesis\...`).  
To run them successfully on your machine:

1. **Create your own folder structure** for raw/preprocessed images.
2. **Update all paths** in the notebooks:
   - Input directories for original images
   - Output directories for masks, isolated iris, normalized images
   - Path to the downloaded model checkpoint
3. **Update paths inside the JSON splits**:
   - Open each `_split.json` file
   - Replace the old absolute paths with your new ones (or use relative paths if possible)
   - Save the modified JSONs

This is required because biometric datasets cannot be shared publicly, and the splits were created on the author's local setup.

## Datasets Used

| Dataset       | Country    | Type       | Source / Access |
|---------------|------------|------------|-----------------|
| CUST-Iris     | Pakistan   | NIR        | Available upon request (contact authors) |
| CASIA-Interval| China      | NIR        | Public (CASIA website) |
| UPOL          | Czech Rep. | Visible    | Public |
| IITD          | India      | NIR        | Public |
| AMF           | Iraq       | NIR        | Public |
| MMU V1        | Malaysia   | Visible    | Public |
| UTIRIS        | Iran       | NIR/Visible| Public |

Download the public datasets, apply the preprocessing notebooks in order (segmentation → isolation → normalization), then use the provided splits (after path correction).

## Requirements

```bash
pip install torch torchvision timm onnxruntime opencv-python tqdm scikit-learn matplotlib pillow numpy
Tested with Python 3.9–3.11 and PyTorch 2.0+.
Citation
If you use this code, the pipeline, or the CUST-Iris dataset, please cite:
@article{khan2026fedhat,
  title   = {FedHAT: Federated Heterogeneity-Aware Training for Cross-Country Iris Verification},
  author  = {Khan, Owais Ali and Taj, Imtiaz Ahmed},
  journal = {Expert Systems with Applications},
  year    = {2026},
}
License
This project is licensed under the MIT License — see LICENSE for details.
Contact
Owais Ali Khan
Email: awais.ali.khan610@gmail.com

Imtiaz Ahmed Taj
Email: iataj777@gmail.com

For access to the CUST-Iris dataset or any questions, feel free to reach out.
