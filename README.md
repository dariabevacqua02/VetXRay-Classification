# VetXRay-Classification
## Deep Learning for Veterinary X-Rays: Disease and Breed Classification

## Overview
This repository contains the code and methodology for my Thesis in Data Science. The project develops deep learning pipelines for veterinary radiography to address complex multi-label (disease) and multi-class (breed) classification problems. 

By analyzing both spatial and frequency domains using a frozen **DINOv2** as a feature extractor, and implementing an **Attention-based Fusion mechanism**, the project demonstrates that integrating spatial and frequency features provides complementary information that significantly enhances overall diagnostic performance.

## Dataset
The raw dataset consists of veterinary DICOM X-Rays. After an extensive EDA and cleaning pipeline (filtering unreadable files, non-informative tags, and rare classes), the final operational dataset contains **7,147 images** featuring:
* **9 Disease Classes** (Multi-label problem: Cardiomegaly, Alveolar Pattern, Pleural Effusion, etc.)
* **10 Breed Classes** (Multi-class problem)
* Both Canine and Feline species.

## Architecture & Methodology
The project is structured into two main phases, tackling both multi-label (disease) and multi-class (breed) classification:

**Phase 1: Baseline Architecture Evaluation**
* Evaluated 4 state-of-the-art deep learning models to establish a robust baseline: **ResNet-50**, **ConvNeXt-Tiny**, **Swin Transformer-Tiny**, and **Vision Transformer (ViT-Base)**.

**Phase 2: Multi-Domain Analysis & Fusion (DINOv2)**
* **Spatial & Frequency Domains:** Transitioned to a frozen **DINOv2** foundation model, using it exclusively as a feature extractor to analyze images independently in their standard spatial domain and their transformed frequency domain (Discrete Fourier Transform).
* **Attention-Based Feature Fusion:** Implemented an attention mechanism to fuse the extracted spatial and frequency representations. This standalone phase successfully demonstrated that the two domains yield complementary information, increasing diagnostic accuracy.

**Core Pipeline Highlights:**
* **Patient-Level Splitting:** Implemented strict 70/15/15 train/val/test splits grouped by `PatientID` to completely eliminate data leakage.
* **DICOM Preprocessing:** Built a robust pipeline to handle missing transfer syntaxes, apply VOI LUT corrections, normalize percentiles, and generate 224x224 cached PNGs for faster training.
* **Data Augmentation:** Leveraged the `Albumentations` library to apply dynamic transformations during training (random rotations, horizontal flips, brightness/contrast adjustments, and Gaussian blur) to improve robustness and prevent overfitting.
* **Handling Class Imbalance:** Computed and applied dynamic class weights for both the multi-label (`pos_weight` in BCE) and multi-class (weighted CrossEntropy).
* **Explainable AI (XAI):** Integrated **Grad-CAM** implemented to visually interpret the models' focus areas for specific pathologies and the bones structure for specific breeds.


## Details
* **Languages & Frameworks:** Python, PyTorch, Torchvision, Timm
* **Image Processing:** OpenCV, Pydicom, Albumentations
* **Data Manipulation & ML:** Pandas, NumPy, Scikit-learn
* **Experiment Tracking:** Weights & Biases (wandb)


