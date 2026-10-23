# Cancer Intelligence: FCM–CNN–Grad-CAM–T-cell model

This project provides a research implementation inspired by the paper:
“Next-Gen Cancer Intelligence: Synergizing Deep Learning with T-Cell Immunotherapy for Precision Oncology”.

## 1. Environment
Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset layout
Arrange images into class folders (the folder names must be `benign` and `malignant`):

```text
dataset/
  benign/
    image001.jpg
    ...
  malignant/
    image002.jpg
    ...
```

Images can be JPG, JPEG, PNG, or BMP. Keep patient-level groups together across splits if patient IDs are available; the simple starter split in this project is image-level and can leak patient-specific patterns if several images come from the same patient.

## 3. Run the programs
From this folder:

```bash
# 1. Create stratified train/validation/test manifests (70/15/15)
python 01_prepare_dataset.py --data dataset --out data_split

# 2. Train the hybrid FCM + CNN model
python 02_train_fcm_cnn.py --split data_split --out runs/fcm_cnn --epochs 100

# 3. Evaluate on the held-out test set
python 03_evaluate.py --split data_split --model runs/fcm_cnn/best_model.keras --out runs/evaluation

# 4. Generate Grad-CAM and FCM segmentation examples
python 04_explain.py --model runs/fcm_cnn/best_model.keras --image dataset/malignant/image002.jpg --out runs/explanations

# 5. Run the illustrative T-cell ODE model
python 05_tcell_response.py --antigen 0.8 --alpha 0.7 --beta 0.15 --days 30 --out runs/tcell_response.csv
```

The training program performs FCM preprocessing on the fly. FCM can be computationally expensive; use `--max-images` for a small smoke test if desired. To use all images, omit it.

## 4. Model details
- Images resized to 224 × 224.
- Three FCM clusters intended to represent background, healthy tissue, and tumor-like regions.
- FCM membership maps are combined with RGB input as a 4th channel.
- A compact residual CNN classifies benign vs malignant.
- Evaluation includes accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and classification report.
- Grad-CAM provides a visual explanation of CNN activation.
- T-cell simulation uses dT/dt = alpha × A × T − beta × T, as stated in the paper.

## 5. Limitations
1. FCM clusters are unsupervised and do not inherently know which cluster is tumor. The code uses a heuristic to select a tumor-like membership map; inspect overlays carefully.
2. The model uses image-level labels only. It does not predict PD-L1, HER2, CD3, antigen specificity, actual treatment response, or patient eligibility.
3. The ODE's antigen input is supplied by the user as a normalized illustrative value, not calibrated from a clinical assay.
4. Do not report paper-reported numbers as reproduced unless obtained in a documented experiment with a matching protocol and dataset.
