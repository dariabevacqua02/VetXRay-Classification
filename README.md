# Deep Learning for Veterinary X-Rays: Disease and Breed Classification with Spatial, Frequency and Fusion Analysis

Code accompanying the Master's thesis in Data Science, University of Catania (A.Y. 2025/2026).

## Dataset
Banzato, T., Burti, S., Zotti, A., & Wodzinski, M. (2026). VetXRay - A Dataset of 9,882 Manually Annotated Canine and Feline Thoracic Radiographs with Lesion and Image Quality Annotations [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.19051776

## Overview

This project studies the automatic classification of canine and feline thoracic radiographs on two tasks:

- **Disease classification**: a multi-label problem over 9 pathologies (cardiomegaly, alveolar pattern, bronchial pattern, pleural effusion, mass, interstitial pattern, pneumothorax, pleural mineralization, megaesophagus).
- **Breed classification**: a multi-class problem over 10 breeds.

Two learning paradigms are compared:

1. **Full fine-tuning** of four ImageNet-pretrained architectures: ResNet-50, ConvNeXt, ViT and Swin.
2. **Frozen self-supervised encoder**: DINOv2 (ViT-S/14) is used as a fixed feature extractor and only a lightweight classifier is trained. Three input configurations are compared: the **spatial** domain (the image), the **frequency** domain (the Discrete Fourier Transform spectrum) and their **fusion**, integrated through an attention mechanism.

The models are further analysed through a ranking-based evaluation of the multi-label predictions and through Grad-CAM.

## Dataset

The experiments use the VetXRay dataset of veterinary thoracic radiographs (DICOM format) with its annotation file. **The data are not included in this repository.** [Add the dataset reference and link here.]

Cleaning pipeline:

| Step | Remaining records |
|---|---|
| Raw annotation file | 9,973 |
| Species restricted to dog and cat | 9,894 |
| Images marked as excluded removed | 7,659 |
| Rare-only or untagged records removed | 7,603 |
| Records with an image file on disk | 7,382 |
| Conflicting duplicate annotations removed | 7,380 |
| Unreadable DICOM files removed | **7,147** |

Breeds are kept when they reach at least 100 cases (10 breeds), and images with an invalid breed label are flagged with `has_breed = 0` and remain available for the disease task.

The train, validation and test splits are **patient-level** (70/15/15, grouped by `PatientID`), so that radiographs of the same animal never appear in different splits. The final set contains 7,147 images from 3,107 patients.

**Preprocessing**: DICOM decoding, VOI LUT, MONOCHROME1 inversion, clipping to the 1st and 99th percentiles, min-max normalization, resize to 224×224 (area interpolation), grayscale replicated to three channels, ImageNet standardization.

## Methods

### Paradigm 1: fine-tuning
- Backbones: ResNet-50, ConvNeXt, ViT, Swin.
- Losses: Binary Cross-Entropy with logits and per-class positive weights for disease; Categorical Cross-Entropy with inverse-frequency class weights for breed.
- AdamW with cosine schedule, early stopping on validation loss (patience 8, max 50 epochs).
- Augmentation: rotation, horizontal flip, brightness and contrast, Gaussian blur.
- Regularization: dropout on the head, weight decay, stochastic depth (timm models).
- Each configuration is trained with **5 seeds** (42, 7, 123, 2024, 99).

### Paradigm 2: frozen DINOv2 with spatial-frequency fusion
- Two frozen DINOv2 encoders: one receives the image, the other its DFT log-magnitude spectrum.
- The two 384-dimensional feature vectors are treated as a sequence of two tokens, combined with a learnable modality embedding and processed by multi-head self-attention (4 heads) with residual connection and LayerNorm.
- Only the fusion module and the linear head are trained.

### Supporting analyses
- **Top-K ranking**: the nine probabilities are sorted and a Hit is counted when at least one true pathology falls within the top K positions.
- **Rank-1 miss analysis**: a reference pathology is fixed among the annotated ones, independently of the model. When the top prediction differs from it, the analysis checks whether it is nonetheless another annotated pathology (co-existing condition) or a true error.
- **Grad-CAM** on ConvNeXt, to verify that predictions rely on anatomical structures rather than acquisition markers.

## Main results (test set, mean over 5 seeds)

| Task | Best configuration | Key metrics |
|---|---|---|
| Disease, fine-tuning | ConvNeXt and ViT (equivalent within seed variability) | AP 0.376 and 0.380 |
| Breed, fine-tuning | Swin | Balanced Accuracy 0.586, AP 0.633 |
| Disease, frozen DINOv2 | Fusion | AP 0.233 (spatial 0.206, frequency 0.116) |
| Breed, frozen DINOv2 | Fusion | Balanced Accuracy 0.470, AP 0.498 (spatial 0.428, frequency 0.203) |

Ranking analysis (disease): Accuracy@1 = 0.609, @2 = 0.790, @3 = 0.880. Of the 297 Rank-1 misses over 585 images, 74 (24.9%) correspond to a co-existing pathology, raising the accuracy from 0.492 to 0.619.

Main findings:
- The fusion of spatial and frequency representations outperforms each single domain on both tasks.
- Fine-tuned models overfit early, and regularization reduces overfitting without improving test performance, which points to dataset size as the main limiting factor.
- The frozen-encoder setting shows much less overfitting.
- Accuracy is inflated by class imbalance, so AP, AUC and Balanced Accuracy are the reference metrics.




## References

- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Liu et al., *A ConvNet for the 2020s*, CVPR 2022.
- Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021.
- Liu et al., *Swin Transformer*, ICCV 2021.
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024.
- Selvaraju et al., *Grad-CAM*, ICCV 2017.
- Banzato et al., *Automatic classification of canine thoracic radiographs using deep learning*, Scientific Reports, 2021.

## Author

Daria Bevacqua, Master's Degree in Data Science, University of Catania.
Professor Luca Guarnera, Academic supervisor, University of Catania.
Professor Sebastiano Battiato, Co-supervisor, University of Catania.
Dr. Riccardo Raciti, Co-supervisor, University of Catania.

## License

The code in this repository is released under the [MIT License](LICENSE).
The VetXRay dataset is not covered by this license: it is distributed separately under the Creative Commons Attribution 4.0 International License (CC BY 4.0).





