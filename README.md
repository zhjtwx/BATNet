# BATNet: Symmetry-Aware Deep Learning for Brown Adipose Tissue Detection

BATNet is a cascaded deep learning framework for **brown adipose tissue (BAT)** detection from chest CT. It consists of:

- **MG_ATSeg**: multi-slice adipose tissue segmentation with a mirror-attention (symmetry-aware) U-Net.
- **CE_BATCIs**: contralateral interaction + transformer aggregation for case-level BAT classification.



## Table of Contents
1. [1 Quick Start](#1-quick-start)
2. [2 Inference](#2-inference)
3. [3 Training](#3-training)
4. [4 Model Architecture](#4-model-architecture)

## 1 Quick Start
### 1.1 BATNet: Quick Start Guide
BATNet is a cascaded deep learning framework for automated brown adipose tissue (BAT) detection from chest CT. It consists of two sequential stages:
- **MG_ATSeg**  
  A symmetry-aware segmentation network for adipose tissue segmentation.
- **CE_BATCIs**  
  A contralateral interaction classification network that predicts BAT probability from bilateral adipose patches.
The complete inference workflow is illustrated below:

```text
Chest CT + Lung Mask -> MG_ATSeg Segmentation -> Bilateral Patch Extraction -> CE_BATCIs Classification -> BAT Probability
```
BATNet now provides two complementary inference modes: **end-to-end inference** (from raw CT and lung masks) and **classification-only inference** (from pre-extracted adipose patches). To facilitate reproducibility and rapid validation, we also provide two fully supported deployment options—**Docker** and **Conda**—for both rapid validation and demo training scripts. In addition, lightweight demonstration datasets are included, allowing users to quickly verify the complete BATNet pipeline, including MG_ATSeg segmentation and CE_BATCIs classification.

#### Choose one of the following installation methods: 

- [🐳 1.2 Docker (Recommended)](#-12-docker-recommended)
- [🐍 1.3 Conda Installation](#-13-conda-installation)


### 🐳 1.2 Docker (Recommended)
We provide a pre-configured Docker image for rapid validation and reproducibility. The Docker image already contains the BATNet source code, model weights, and validation datasets, so it can run the examples without downloading separate files on the host machine.
> **Note:** The README included in the Docker image may differ from the latest version on GitHub. Please refer to the GitHub repository for the most up-to-date documentation and usage instructions.
#### 1.2.1 Prerequisites (host machine)
- A Linux host is recommended (Ubuntu 22.04 or later).
- End-to-end inference requires a CUDA-capable GPU.
- Ensure **Docker + NVIDIA Container Toolkit** are installed and that `docker run --gpus all ...` works on your machine.

#### 1.2.2 Download & load the image

- **Docker image archive**: [https://zenodo.org/records/19837616/files/batnet-image-v3.tar.gz?download=1](https://zenodo.org/records/19886391/files/batnet-image-v3.tar.gz?download=1)
- Total download size: 15.6 GB

```bash
docker load -i batnet-image-v3.tar.gz
docker images | grep batnet-image
```

#### 1.2.3 Run the container

There are two common ways to run BATNet in Docker:

##### Option A: start a shell, then run inference inside

```bash
docker run --gpus all -it --rm --shm-size=16g batnet-image:v3 /bin/bash
```
After startup, you will automatically enter:

```bash
root@container:/BATNet#
```
##### Run Inference
```bash
# Classification-only inference using preprocessed patches (744 cases)
python bat_inf.py --input data/crop_744/info.csv --output crop_744_inf.csv
# End-to-end inference from complete CT scans (10 cases)
python bat_inf.py --input data/nii_10 --output nii_10_inf.csv
```
##### Run Training
```bash
# MG_ATSeg (segmentation)
python MG_ATSeg/train.py
# CE_BATCIs (classification)
python CE_BATCIs/ce_train.py
```

#### Option B: run inference directly from docker run

##### Run Inference
```bash
docker run --gpus all --rm --shm-size=16g batnet-image:v3 \
  bash -lc "python bat_inf.py --input data/nii_10 --output nii_10_inf.csv"
```
```bash
docker run --gpus all --rm --shm-size=16g batnet-image:v3 \
  bash -lc "python bat_inf.py --input data/crop_744/info.csv --output crop_744_inf.csv"
```

##### Run Training
```bash
docker run --gpus all --rm --shm-size=16g batnet-image:v3 \
  bash -lc "python MG_ATSeg/train.py"
```
```bash
docker run --gpus all --rm  --shm-size=16g batnet-image:v3 \
  bash -lc "python CE_BATCIs/ce_train.py"
```
### 🐍 1.3 Conda Installation
#### 1.3.1 Install dependencies
To improve compatibility, security, and long-term maintainability, we have upgraded the BATNet runtime environment. The original release was developed under an earlier software stack, while the updated release adopts a modern and actively maintained environment. Both versions have been extensively validated and are fully functional. Users may choose either environment according to their hardware and software requirements.
##### Environment Comparison
| Component | Original Environment | Updated Environment |
|-----------|--------------|--------------|
| Operating System| Ubuntu 16.04.4 LTS| Ubuntu 22.04.2 LTS|
| Python | Python 3.6.13| Python 3.11|
| PyTorch | PyTorch 1.10.0+cu113| PyTorch: 2.4.1+cu121|
| Status| Fully Supported| Fully Supported|
| Test AUC (744 cases) | 0.896| 0.896|
##### Reproducibility Verification
Both the original and updated environments have been extensively validated. On the independent test cohort of 744 cases, both versions achieved an identical AUC of 0.896. In addition, we tested the model on multiple GPU platforms, including the NVIDIA GeForce RTX 3090 and the NVIDIA Tesla V100-SXM2-32GB. While minor numerical differences may occur in the predicted probabilities across different GPU architectures (typically at the fifth decimal place), the overall evaluation metrics—including AUC, sensitivity, and specificity—remain unchanged. 

##### Updated Environment
```bash
# Clone repository
git clone --recurse-submodules https://github.com/zhjtwx/BATNet.git
cd BATNet
# Install core packages
# 1.Create Conda Environment
conda create -n batnet python=3.11 -y
conda activate batnet
# 2.Install PyTorch with CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
# 3.Install Other
pip install -r requirements.txt
```
##### Original Environment
```bash
# Clone repository
git clone --recurse-submodules https://github.com/zhjtwx/BATNet.git
cd BATNet
# Install core packages
# 1.Create Conda Environment
conda create -n batnet python=3.6.13 -y
conda activate batnet
# 2.Install PyTorch with CUDA 11.3
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# 3.Install Other
pip install -r requirements_old.txt
```
##### Expected Install Time
The installation typically takes:
- **15-25 minutes** on a normal desktop with good internet connection
- May extend to **30-45 minutes** if compilation is required for your specific system
#### 1.3.2 Download model weights
Download from Zenodo and extract into the repository root:
- **Weights**: https://zenodo.org/records/17540420/files/model_weights.zip?download=1
- **Password**: batnet
  
Expected structure:
```bash
BATNet/
├── model_weights/
│   ├── ce_batcis_model.pth
│   ├── mg_atseg_model.pth
│   └── ...
├── bat_inf.py
├── pro_at_patch.py
└── ...
```
#### 1.3.3 Download validation data (optional but recommended)
We provide two complementary validation datasets:
- **Adipose patch dataset (n = 744)**: classification-only (pre-extracted bilateral patches)
- **Complete CT dataset (n = 10)**: end-to-end inference (CT + lung mask)
Download and extract into `BATNet/data/`:
- **Data**: https://zenodo.org/records/17541503/files/data.zip?download=1
  **Password**: batnet

Example directory layout:
```bash
BATNet/
└── data/
    ├── crop_744/
    │   ├── info.csv
    │   ├── case_0001/
    │   │   ├── ct_at_left_patch.nii.gz
    │   │   └── ct_at_right_patch.nii.gz
    │   └── ...
    └── nii_10/
        ├── info.csv
        ├── case_0001/
        │   ├── image.nii.gz
        │   └── lobe.nii.gz
        └── ...
```
#### 1.3.4 Download Sample Training Data
To help users quickly verify the training pipeline, we provide a lightweight demonstration dataset. 3 cases for MG_ATSeg training and 5 cases for CE_BATCIs training. 
Download and extract into `BATNet/data/`:
- **Data**: https://zenodo.org/records/15524145/files/data.zip?download=1.
Example directory layout:
```bash
BATNet/
└── data/
    ├── seg_data/
    │   ├── train/
    │   └── test/
    └── cls_data/
        ├── train/
        └── test/
```
##### MG_ATSeg Dataset Structure
```bash
data/
└── seg_data/
    ├── train/
    │   ├── case_001/
    │   │   ├── data_png/
    │   │   └── data_mask/
    │   └── case_002/
    └── test/
```
##### CE_BATCIs Dataset Structure
```bash
data/
└── cls_data/
    ├── train/
    │   ├── case_001/
    │   └── case_002/
    └── test/
```
#### 1.3.5 Run inference

```bash
# Classification-only inference using preprocessed patches (744 cases)
python bat_inf.py --input data/crop_744/info.csv --output crop_744_inf.csv

# End-to-end inference from complete CT scans (10 cases)
python bat_inf.py --input data/nii_10 --output nii_10_inf.csv
```

The output CSV contains:

- `case_id`: absolute or original case directory path
- `prediction`: probability for the positive class

When running end-to-end inference (CT + lung mask), BATNet additionally saves:

- `seg_at_mask.nii.gz`: adipose tissue mask (written into each case directory)

#### 1.3.6 Run train
```bash
# MG_ATSeg (segmentation)
python MG_ATSeg/train.py
# CE_BATCIs (classification)
python CE_BATCIs/ce_train.py
```
---
For detailed inference, and training instructions, please refer to the corresponding sections below:
- **Inference Details**: [2 Inference](#2-inference)
- **Training Details**: [3 Training](#3-training)
- **Model Architecture**: [4 Model Architecture](#4-model-architecture)
---

## 2 Inference
BATNet is a cascaded deep learning framework consisting of two sequential modules: an adipose tissue segmentation model and a brown adipose tissue (BAT) classification model. In this updated release, we have added a new unified inference script, bat_inf.py, which provides a streamlined and flexible interface for model deployment. The script supports multiple input formats and automatically selects the appropriate inference pipeline based on the provided data, making BATNet significantly easier to use for both end-to-end prediction and rapid evaluation of preprocessed cases.

---
### 2.1 Validation Datasets
We provide two complementary validation datasets:
- **Adipose Patch Dataset (n = 744)**
This dataset contains preprocessed bilateral adipose patches extracted from the independent holdout cohort. It is designed for validating the BAT classification module only.
- **Complete CT Dataset (n = 10)**
This dataset contains complete chest CT scans in NIfTI format together with their corresponding lung masks. It enables end-to-end validation of the full BATNet pipeline, including adipose tissue segmentation, patch extraction, and BAT classification.
---

#### 2.1.1 Directory Structure of Validation Datasets
```bash
BATNet/
├── data/
│     ├── crop_744/
│     │   ├── info.csv
│     │   ├── case_0001/
│     │   │   ├── ct_at_left_patch.nii.gz
│     │   │   └── ct_at_right_patch.nii.gz
│     │   ├── case_0002/
│     │   │   ├── ct_at_left_patch.nii.gz
│     │   │   └── ct_at_right_patch.nii.gz
│     │   └── ...
│     │
│     └── nii_10/
│         ├── info.csv
│         ├── case_0001/
│         │   ├── image.nii.gz
│         │   └── lobe.nii.gz
│         ├── case_0002/
│         │   ├── image.nii.gz
│         │   └── lobe.nii.gz
│         └── ...
├── bat_inf.py
├── pro_at_patch.py
└── ...
```
---

##### Complete CT Dataset (nii_10)
This format is intended for fully automated, end-to-end BATNet inference.
```bash
case_xxxx/
├── image.nii.gz
└── lobe.nii.gz
```
- **image.nii.gz**  
  Chest CT volume converted directly from the original DICOM series.
- **lobe.nii.gz** 
  Binary lung mask generated using an automated lung segmentation tool, such as TotalSegmentator.
This dataset is used for fully automated end-to-end inference, including adipose tissue segmentation, patch extraction, and BAT classification.
##### Adipose Patch Dataset (`crop_744`)
This format is intended for rapid BAT classification and matches the provided crop_744 validation dataset.
```bash
case_xxxx/
├── ct_at_left_patch.nii.gz
└── ct_at_right_patch.nii.gz
```
- **ct_at_left_patch.nii.gz**  
  Cropped left-sided adipose tissue volume encompassing the axillary region and all superior anatomical regions, automatically extracted from the chest CT.

- **ct_at_right_patch.nii.gz**  
  Cropped right-sided adipose tissue volume encompassing the axillary region and all superior anatomical regions, automatically extracted from the chest CT.

These bilateral adipose patches are generated using `pro_at_patch.py` and serve as direct input to the CE_BATCIs classification network.
This dataset is specifically designed for rapid validation of the BAT classification module, without requiring adipose tissue segmentation or patch extraction.

---

#### 2.1.2 Preparing Your Own Data

##### Option 1: Direct End-to-End Inference
Prepare your data in the Complete CT format shown above:
```bash
case_xxxx/
├── image.nii.gz
└── lobe.nii.gz
```
This format is fully compatible with the provided nii_10 validation dataset.

---
##### Option 2: Generate Adipose Patches for Rapid Classification
To generate BAT classification patches compatible with the provided crop_744 dataset, prepare the following files:
```bash
case_xxxx/
├── image.nii.gz
├── lobe.nii.gz
└── seg_at_mask.nii.gz
```
- image.nii.gz: Original chest CT volume.
- lobe.nii.gz: Binary lung mask.
- seg_at_mask.nii.gz: Binary adipose tissue mask obtained by:Manual annotation, or Automatic adipose tissue segmentation.
###### Patch Extraction
After preparation, use SymmetricalATProcessor to generate bilateral adipose patches.
```python
import sys
sys.path.append('.')

from pro_at_patch import SymmetricalATProcessor, save_nii

processor = SymmetricalATProcessor(
    model_file="./model_weights/mg_atseg_model.pth"
)

ct_at_left_patch, ct_at_right_patch, _, _ = processor.process(
    image_file,   # Path to the CT image.('./data/nii_10/case_20462/image.nii.gz')
    fat_file,     # Path to the adipose tissue mask.('./data/nii_10/case_20462/seg_at_mask.nii.gz')
    None,
    lung_file     # Path to the lung mask. ('./data/nii_10/case_20462/lobe.nii.gz')
)

save_nii(
    ct_at_left_patch,
    './data/nii_10_test/case_20462/ct_at_left_patch.nii.gz'
)

save_nii(
    ct_at_right_patch,
    './data/nii_10_test/case_20462/ct_at_right_patch.nii.gz'
)
```
###### Generated Output

The generated files follow exactly the same structure as the provided crop_744 validation dataset:
```bash
case_xxxx/
├── ct_at_left_patch.nii.gz
└── ct_at_right_patch.nii.gz
```
These files can then be directly used for BATNet classification.

---

##### Important Note
All mask files must be stored as binary masks:

- 1 indicates foreground.
- 0 indicates background.

This requirement applies to the following files:

- lobe.nii.gz
- seg_at_mask.nii.gz
---

### 2.2 Inference details 
BATNet provides a unified inference script, `bat_inf.py`, which automatically detects the input format and selects the appropriate processing pipeline.

- For **Complete CT cases** (`image.nii.gz` + `lobe.nii.gz`), BATNet performs:
  1. Adipose tissue segmentation
  2. Bilateral patch extraction
  3. BAT classification

- For **Adipose Patch cases** (`ct_at_left_patch.nii.gz` + `ct_at_right_patch.nii.gz`), BATNet directly performs BAT classification.

This unified design allows both provided validation datasets (`nii_10` and `crop_744`) and your own data to be processed using the same command.

---

#### 2.2.1 bat_inf.py
###### Detailed Usage
```bash
python bat_inf.py \
    --input <input_path> \
    --cls_weight ./model_weights/ce_batcis_model.pth \
    --seg_weight ./model_weights/mg_atseg_model.pth \
    --device cuda \
    --batch_size 32 \
    --num_workers 6 \
    --output results.csv
```
---

###### Parameter Explanation

---

- --input (**Required**)
Input source. BATNet supports three formats:
1. CSV File: The CSV must contain a case_id column listing case directory paths.
```bash
python bat_inf.py --input data/crop_744/info.csv --output crop_744_results.csv
```
2. Root Directory: BATNet recursively scans all subdirectories and automatically identifies valid cases.
```bash
python bat_inf.py --input data/nii_10 --output nii_10_results.csv
```
3. Multiple Case Directories
```bash
python bat_inf.py --input data/nii_10/case_20076 data/nii_10/case_20195 --output selected_cases.csv
```
**Valid Case Formats**

A directory is considered valid if it satisfies either of the following:
- Complete CT Format
```bash
case_xxxx/
├── image.nii.gz
└── lobe.nii.gz
```
- Adipose Patch Format
```bash
case_xxxx/
├── ct_at_left_patch.nii.gz
└── ct_at_right_patch.nii.gz
```
---
- --cls_weight: BAT classification model weight path Default: ./model_weights/ce_batcis_model.pth.
---
- --seg_weight: Path to the adipose tissue segmentation model. Default: ./model_weights/mg_atseg_model.pth. (only for end-to-end)
---
- --device: cuda (default) / cpu; end-to-end inference only supports CUDA.
> **Important:** For end-to-end inference, the segmentation stage is configured to run on **GPU 0**. Please ensure that GPU 0 is available before execution.
---

- --batch_size: Classification batch size, adjust according to GPU memory
---
- --num_workers: DataLoader multi-process workers

---

- --output

Optional output CSV file. If omitted, results are printed to the terminal only. The generated CSV contains:

| case_id | prediction |
|-----------|--------------|
| ./data/nii_10/case_20076 | 0.91093576|
| ./data/nii_10/case_20195 | 0.04492249|

case_id: Absolute case directory path; prediction: BAT probability score. Higher scores indicate a greater likelihood of BAT positivity.



## 3 Training
BATNet includes reference training scripts for both stages:

- **MG_ATSeg**: `MG_ATSeg/train.py`
- **CE_BATCIs**: `CE_BATCIs/ce_train.py`

> Important: The provided training scripts are intended as **reference / demonstration pipelines**. They use fixed default paths under `./data/` and may require adaptation for custom datasets (e.g., proper train/val/test splits, logging, and checkpoint management).
> The demo scripts are not full experiment-management pipelines: the default segmentation script uses a single demo case for train/validation/test, and the default classification script builds train and validation loaders from the same `./data/cls_data/train/` directory. Update these paths and splits before using the scripts for formal experiments.

---
### 3.1 Sample Training Data (demo)

To help users quickly verify the training pipeline, we provide a lightweight demonstration dataset.
- 3 cases for MG_ATSeg training
- 5 cases for CE_BATCIs training

If you have already downloaded the Quick Start package, no additional download is required.
After downloading, extract the archive into the BATNet root directory.
#### 3.1.1 Directory Structure
```bash
BATNet/
└── data/
    ├── seg_data/
    │   ├── train/
    │   └── test/
    └── cls_data/
        ├── train/
        └── test/
```
#### 3.1.2 MG_ATSeg Dataset Structure
```bash
data/
└── seg_data/
    ├── train/
    │   ├── case_001/
    │   │   ├── data_png/
    │   │   └── data_mask/
    │   └── case_002/
    └── test/
```
##### File Description
- data_png/
   - Input CT slices in PNG format.
- data_mask/
  - Corresponding adipose tissue segmentation masks.
Each PNG image must have a matching mask file with the same filename.
##### Mask Format
The adipose tissue segmentation mask must be stored as an 8-bit binary PNG image:
- Foreground (adipose tissue): 255
- Background: 0
Each mask filename must exactly match its corresponding input image.
For example:
```bash
data_png/15.png
data_mask/15.png
```
###### Preprocessing Pipeline
Starting from the original DICOM series, each axial slice undergoes the following preprocessing steps:
- 1. Convert the DICOM volume to NIfTI format.
- 2. Apply Hounsfield Unit (HU) windowing:
     - Lower bound: -300 HU
     - Upper bound: 200 HU
- 3. Normalize intensities to the range [0, 255].

Save each axial slice as an 8-bit grayscale PNG image.
This preprocessing enhances adipose tissue contrast while suppressing irrelevant structures.

---

#### 3.1.3 CE_BATCIs Dataset Structure
```bash
data/
└── cls_data/
    ├── train/
    │   ├── case_001/
    │   └── case_002/
    └── test/
```
Each case directory should contain:
```bash
case_xxxx/
├── image.nii.gz
├── fat_mask.nii.gz
├── brown_fat_mask.nii.gz
├── lobe.nii.gz
├── ct_at_left_label.nii.gz
├── ct_at_right_label.nii.gz
├── ct_at_left_patch.nii.gz
└── ct_at_right_patch.nii.gz
```
##### File Description
- image.nii.gz: Full chest CT volume in NIfTI format.
- fat_mask.nii.gz: Adipose tissue mask.
- brown_fat_mask.nii.gz: Ground-truth brown adipose tissue annotation.
- lobe.nii.gz: Lung mask. Used to extract adipose tissue above the axillary region while excluding inferior adipose tissue.
- ct_at_left_patch.nii.gz: Left axillary adipose patch.
- ct_at_right_patch.nii.gz: Right axillary adipose patch.
- ct_at_left_label.nii.gz: Ground-truth BAT label for the left patch.
- ct_at_right_label.nii.gz: Ground-truth BAT label for the right patch.

The patch and label files can either be generated offline using pro_at_patch.py or automatically created during training.
##### CE_BATCIs Data Preparation
BATNet supports two flexible data preparation strategies for CE_BATCIs training.
###### Option 1: End-to-End Online Patch Generation (Recommended)
Users only need to prepare the following four files:
```bash
case_xxxx/
├── image.nii.gz
├── fat_mask.nii.gz
├── brown_fat_mask.nii.gz
└── lobe.nii.gz
```
###### Required Files
- image.nii.gz: Original chest CT volume in NIfTI format.
- fat_mask.nii.gz: Adipose tissue mask. If this file is unavailable, BATNet will automatically invoke the MG_ATSeg model to generate it during training.
- brown_fat_mask.nii.gz: Ground-truth brown adipose tissue annotation. This file is optional but strongly recommended for supervised training.
- lobe.nii.gz: Lung mask.
  
During training, BATNet automatically performs:
- Bilateral adipose patch extraction
- Label generation
- End-to-end CE_BATCIs optimization
  
The following files will be generated online:
```bash
├── ct_at_left_patch.nii.gz
├── ct_at_right_patch.nii.gz
├── ct_at_left_label.nii.gz
└── ct_at_right_label.nii.gz
```
This is the simplest and recommended workflow for most users.

###### Option 2: Offline Patch Generation
Alternatively, users may precompute the bilateral adipose patches and corresponding labels before training. This approach is recommended for large-scale experiments, as it significantly accelerates training.
###### Required Input Files
```bash
case_xxxx/
├── image.nii.gz
├── fat_mask.nii.gz
├── brown_fat_mask.nii.gz
└── lobe.nii.gz
```
###### Patch Generation Example
```python
import sys
sys.path.append('.')
import numpy as np
import nibabel as nib
from pro_at_patch import SymmetricalATProcessor, save_nii

# Initialize processor
processor = SymmetricalATProcessor(
    model_file="./model_weights/mg_atseg_model.pth"
)

# Generate bilateral adipose patches
ct_at_left_patch, ct_at_right_patch, \
ct_at_left_label, ct_at_right_label = processor.process(
    image_file,   # ./data/cls_data/train/case_001/image.nii.gz
    fat_file,     # ./data/cls_data/train/case_001/fat_mask.nii.gz
    bat_file,     # ./data/cls_data/train/case_001/brown_fat_mask.nii.gz
    lung_file     # ./data/cls_data/train/case_001/lobe.nii.gz, Can None
)

# Save outputs
nib.save(nib.Nifti1Image(ct_at_left_patch, np.eye(4)), './data/cls_data/train/case_001/ct_at_left_patch.nii.gz')
nib.save(nib.Nifti1Image(ct_at_right_patch, np.eye(4)),  './data/cls_data/train/case_001/ct_at_right_patch.nii.gz')
nib.save(nib.Nifti1Image(ct_at_left_label.astype(np.uint8), np.eye(4)), './data/cls_data/train/case_001/ct_at_left_label.nii.gz')
nib.save(nib.Nifti1Image(ct_at_right_label.astype(np.uint8), np.eye(4)), './data/cls_data/train/case_001/ct_at_right_label.nii.gz')

```
###### Generated Files
```bash
case_xxxx/
├── ct_at_left_patch.nii.gz
├── ct_at_right_patch.nii.gz
├── ct_at_left_label.nii.gz
└── ct_at_right_label.nii.gz
```
These precomputed files can then be directly used by CE_BATCIs without requiring online patch extraction.
##### Recommendation
- For quick experiments or small datasets, use Option 1.
- For large-scale training or repeated experiments, use Option 2 to improve efficiency and reduce preprocessing overhead.

---
### 3.2 Training Commands
#### 3.2.1 MG_ATSeg Training
```bash
python MG_ATSeg/train.py
```
#### 3.2.2 CE_BATCIs Training
For simplicity, we demonstrate the end-to-end training pipeline, where adipose segmentation, patch extraction, and BAT classification are performed automatically.
```bash
python CE_BATCIs/ce_train.py
```
#### Notes
- MG_ATSeg must be trained before CE_BATCIs if adipose masks are not already available.
- During CE_BATCIs training, missing adipose masks and adipose patches will be automatically generated when necessary. By default, the training script requires pre-existing adipose masks. If you need to generate adipose patches on the fly, you must explicitly provide the path to the adipose segmentation model file (mg_atseg_model.pth) via the seg_model_file argument when initializing BATDataset (defined in ce_dataset.py) in the training script CE_BATCIs/ce_train.py, and update the script configuration accordingly.
- For large-scale experiments, we recommend precomputing all patches offline using pro_at_patch.py to accelerate training.

---

## 4 Model Architecture
![BATNet Workflow](model.png) For details of the model structure, please refer to the paper
### 4.1 MG_ATSeg Components

| Module | Description | Location |
|-----------|--------------|-------------|
| MAM |   Mirror Attention Module | MG_ATSeg/model/unet_mam.py |

### 4.2 CE_BATCIs Components

| Module | Description | Location |
|-----------|--------------|-------------|
| CIM |   Contralateral Interaction | CE_BATCIs/model/resnet_cim.py |
| Patch Processing Module |   Symmetrical AT Patch Processing Module | pro_at_patch.py |
| Transformer |   Case-Level Prediction | CE_BATCIs/model/resnet_cim.py |


### 4.3 Key Features
#### 4.3.1 Segmentation Stage (MG_ATSeg)
- **Mirror Attention Module** enforces anatomical symmetry
- **Multi-slice input** (5 consecutive slices)
- **Dice + BCE loss** for robust training

#### 4.3.2 Classification Stage (CE_BATCIs) 
- **Contralateral Interaction Module** compares left-right patches
- **Hybrid ResNet-Transformer** architecture
- **Three-class patch labeling**:
  - Positive (>10% BAT voxels)
  - Negative (0% BAT)
  - Ignored (0-10% BAT)


