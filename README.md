# VetXRay-Classification
## Deep Learning for Veterinary X-Rays: Disease and Breed Classification

## Overview
This repository contains the code and methodology for my Thesis in Data Science. The project develops deep learning pipelines for veterinary radiography to address complex multi-label (disease) and multi-class (breed) classification problems. 

By analyzing both spatial and frequency domains using a frozen **DINOv2** as a feature extractor, and implementing an **Attention-based Fusion mechanism**, the project demonstrates that integrating spatial and frequency features provides complementary information that significantly enhances overall diagnostic performance.

## Architecture & Methodology
The project evaluates 4 state-of-the-art vision models:
* **ResNet-50**
* **ConvNeXt-Tiny**
* **Swin Transformer-Tiny**
* **Vision Transformer (ViT-Base)**

**Key Methodological Highlights:**
* **Patient-Level Splitting:** Implemented strict 70/15/15 train/val/test splits grouped by `PatientID` and `StudyUID` to completely eliminate data leakage.
* **DICOM Preprocessing:** Built a robust pipeline to handle missing transfer syntaxes, apply VOI LUT corrections, normalize percentiles, and generate 224x224 cached PNGs for faster training.
* **Handling Class Imbalance:** Applied dynamic `pos_weight` in `BCEWithLogitsLoss` to tackle the severe imbalance in the 9 disease classes.
* **Explainable AI (XAI):** Integrated **Grad-CAM** hooks on the final convolutional stages to visually interpret the model's focus areas for specific pathologies (e.g., Cardiomegaly).

## Dataset
The raw dataset consists of veterinary DICOM X-Rays. After an extensive EDA and cleaning pipeline (filtering unreadable files, non-informative tags, and rare classes), the final operational dataset contains **7,147 images** featuring:
* **9 Disease Classes** (Multi-label problem: Cardiomegaly, Alveolar Pattern, Pleural Effusion, etc.)
* **10 Breed Classes** (Multi-class problem)
* Both Canine and Feline species.


## Details
* **Languages & Frameworks:** Python, PyTorch, Torchvision, Timm
* **Image Processing:** OpenCV, Pydicom, Albumentations
* **Data Manipulation & ML:** Pandas, NumPy, Scikit-learn
* **Experiment Tracking:** Weights & Biases (wandb)

## Repository Structure
```text
vet-xray-classification/
├── data/                  # Sample data and metadata CSVs
├── notebooks/             # EDA, visualization, and XAI experiments
├── src/                   # Source code (preprocessing, models, training loops)
│   ├── config.py          # Centralized hyperparameters and paths
│   ├── data_prep.py       # DICOM extraction and cleaning pipeline
│   ├── dataset.py         # PyTorch Dataset class and augmentations
│   ├── models.py          # Encoders and Classification Head architectures
│   └── train.py           # Main training loop with W&B integration
├── requirements.txt       # Project dependencies
└── README.md
