# BATNet: Symmetry-Aware Deep Learning for Brown Adipose Tissue Detection

BATNet is a cascaded deep learning framework for **brown adipose tissue (BAT)** detection from chest CT. It consists of:

- **MG-ATSeg**: multi-slice adipose tissue segmentation with a mirror-attention (symmetry-aware) U-Net.
- **CE-BATCls**: contralateral interaction + transformer aggregation for case-level BAT classification.


![BATNet Workflow](model.png) For details of the model structure, please refer to the paper

## Table of Contents
1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Inference](#Inference)
4. [Training](#training) 
5. [Model Architecture](#model-architecture)


## Key Features

### Segmentation Stage (MG-ATSeg)
- **Mirror Attention Module** enforces anatomical symmetry
- **Multi-slice input** (5 consecutive slices)
- **Dice + BCE loss** for robust training

### Classification Stage (CE-BATCls) 
- **Contralateral Interaction Module** compares left-right patches
- **Hybrid ResNet-Transformer** architecture
- **Three-class patch labeling**:
  - Positive (>10% BAT voxels)
  - Negative (0% BAT)
  - Ignored (0-10% BAT)
  
## Installation

### Environment
To improve compatibility, security, and long-term maintainability, we have upgraded the BATNet runtime environment. The original release was developed under an earlier software stack, while the updated release adopts a modern and actively maintained environment. Both versions have been fully validated and are fully functional. Users may choose either environment according to their hardware and software requirements.
#### Environment Comparison
| Component | Original Environment | Updated Environment |
|-----------|--------------|--------------|
| Operating System| Ubuntu 16.04.4 LTS| Ubuntu 22.04.2 LTS|
| Python | Python 3.6.13| Python 3.11|
| PyTorch | Pytorch 1.10.0+cu113| PyTorch: 2.4.1+cu121|
| Status| Fully Supported| Fully Supported|
| Test AUC (744 cases) | 0.896| 0.896|

#### Reproducibility Verification

Both the original and updated environments have been extensively validated. On the independent test cohort of 744 cases, both versions achieved an identical AUC of 0.891. In addition, we tested the model on multiple GPU platforms, including the NVIDIA GeForce RTX 3090 and the NVIDIA Tesla V100-SXM2-32GB. While minor numerical differences may occur in the predicted probabilities across different GPU architectures (typically at the fifth decimal place), the overall evaluation metrics—including AUC, sensitivity, and specificity—remain unchanged. 

### installation steps
#### Updated Environment
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
#### Original Environment
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
#### Expected Install Time
The installation typically takes:
- **15-25 minutes** on a normal desktop with good internet connection
- May extend to **30-45 minutes** if compilation is required for your specific system

### Docker Environment
To facilitate rapid validation and reproducibility, we provide a fully pre-configured Docker environment. The Docker package includes:
- **Complete runtime dependencies**
- **BATNet source code**
- **Example datasets**
- **model weights**

#### Download Link
The Docker image can be downloaded from:[https://zenodo.org/records/19756491/files/batnet-image-v3.tar.gz?download=1] 
#### Quick Start
After downloading the archive, navigate to the directory containing the file and execute the following commands to load the Docker image and launch the BATNet runtime environment:
```bash
docker load -i batnet-image-v3.tar.gz # Load the Docker image after downloading
docker run --gpus all -it --rm --runtime=nvidia --shm-size=16g batnet-image:v3 /bin/bash
```
#### After Launching the Container
Once the container starts, you will automatically enter the BAT-Net project directory:
```bash
root@container:/BATNet#
```
You can then directly follow the inference and training instructions provided below.
The Docker image already includes:
- Model weights for both MG-ATSeg and CE-BATCls
- Two inference demonstration datasets:
  - data/nii_10
  - data/crop_744
- A lightweight training demonstration dataset
  - data/seg_data
  - data/cls_data

This means no additional downloads or configuration are required—everything is ready to use immediately after launching the container.
#### Note! We recommend using Ubuntu 22.04 or later for optimal compatibility and performance.

## Inference
BATNet is a cascaded deep learning framework consisting of two sequential modules: an adipose tissue segmentation model and a brown adipose tissue (BAT) classification model. In this updated release, we have added a new unified inference script, bat_inf.py, which provides a streamlined and flexible interface for model deployment. The script supports multiple input formats and automatically selects the appropriate inference pipeline based on the provided data, making BATNet significantly easier to use for both end-to-end prediction and rapid evaluation of preprocessed cases.

### Model Weights 
The model weights can be downloaded from Zenodo: 
https://zenodo.org/records/17540420/files/model_weights.zip?download=1; Password: batnet
Note: After downloading, please extract the contents to the root directory of the BATNet project.
Password: batnet
The directory structure should look like this:
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
Please ensure that all weight files are placed inside the model_weights/directory. BATNet will automatically locate and load them during inference.

### Validation Datasets
We provide two complementary validation datasets:
#### Adipose Patch Dataset (n = 744)
This dataset contains preprocessed bilateral adipose patches extracted from the independent holdout cohort. It is designed for validating the BAT classification module only.
#### Complete CT Dataset (n = 10)
This dataset contains complete chest CT scans in NIfTI format together with their corresponding lung masks. It enables end-to-end validation of the full BATNet pipeline, including adipose tissue segmentation, patch extraction, and BAT classification.
#### Download
The validation dataset can be downloaded from Zenodo: https://zenodo.org/records/17541503/files/data.zip?download=1;
##### Password: batnet
After downloading and extracting the archive, place the data directory in the BATNet project root.
#### Directory Structure of Validation Datasets
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
### Usage Examples
```bash
# Classification-only inference using preprocessed adipose patches(744 cases)
python bat_inf.py --input data/crop_744/info.csv --output crop_744_inf.csv

# End-to-end inference from complete CT scans (10 cases)
python bat_inf.py --input data/nii_10 --output nii_10_inf.csv
```

#### Input Argument
BATNet provides flexible input handling through the --input argument, allowing users to easily adapt the inference workflow to different data organization styles.
- CSV file
  The CSV file must contain a case_id column, where each row specifies the path to an individual case.
  ``` bash
  python bat_inf.py --input data/crop_744/info.csv --output crop_744_inf.csv
  ```
  The CSV file format can be found in the provided examples: “./data/crop_744/info.csv” and “./data/nii_10/info.csv”

- Single root directory
  BATNet recursively scans all subdirectories under the specified root directory and automatically identifies all valid cases.
  ``` bash
  python bat_inf.py --input data/nii_10 --output nii_10_inf.csv
   ```
  This is the recommended option when all cases are organized under a single parent directory.
- Multiple case directories
  Users may also specify multiple case directories directly.
  ``` bash
  python bat_inf.py \
    --input data/nii_10/case_20565 data/nii_10/case_20591 \
    --output selected_cases_inf.csv
   ```
  Only valid case directories will be processed.
  
#### Output Argument
The --output argument specifies the CSV file used to save prediction results.
``` bash
python bat_inf.py --input data/nii_10 --output nii_10_inf.csv
 ```
Example output:
| case_id | prediction |
|-----------|--------------|
| ./data/nii_10/case_20076 | 0.91093576|
| ./data/nii_10/case_20195 | 0.04492249|

If Input Type 1 (original CT image + lung mask) is used, BATNet will additionally save the segmented adipose mask in the same directory as the original image.
Example output structure:
``` bash
data/nii_10/
└── case_0001/
    ├── image.nii.gz
    ├── lobe.nii.gz
    └── seg_at_mask.nii.gz
 ```
Where: image.nii.gz — Original CT image. lobe.nii.gz — Input lung mask. seg_at_mask.nii.gz — Automatically generated adipose tissue segmentation mask
This additional output is only generated during end-to-end inference.

### Preparing Custom Input Data
BATNet supports two input formats during inference. In addition to the provided validation datasets, users may prepare their own data following the guidelines below.
### 1. Complete CT Dataset
This format corresponds to the provided nii_10 dataset and is intended for fully automated, end-to-end inference.
#### Required Files
``` bash
case_xxxx/
├── image.nii.gz
└── lobe.nii.gz
 ```
- image.nii.gz: Generated by converting the original DICOM series directly into NIfTI format.
- lobe.nii.gz: Lung mask generated using an off-the-shelf lung segmentation model, such as TotalSegmentator.
### 2. Adipose Patch Dataset
This format corresponds to the provided crop_744 dataset and is intended for rapid BAT classification.
#### Required Files
``` bash
├── image.nii.gz
├── lobe.nii.gz
└── seg_at_mask.nii.gz
 ```
- image.nii.gz:Generated by converting the original DICOM series directly into NIfTI format.
- lobe.nii.gz: Lung mask generated using a lung segmentation model such as TotalSegmentator.
- seg_at_mask.nii.gz Adipose tissue mask. This can be obtained either by: Manual annotation, or Automatic prediction using an adipose tissue segmentation model.

After preparing these three files, bilateral adipose patches can be generated using the SymmetricalATProcessor provided in pro_at_patch.py.
#### Example
``` bash
import sys
sys.path.append('.')

from pro_at_patch import SymmetricalATProcessor, save_nii

# Initialize processor
processor = SymmetricalATProcessor(
    model_file="./model_weights/mg_atseg_model.pth"
)

# Generate bilateral adipose patches
ct_at_left_patch, ct_at_right_patch, _, _ = processor.process(
    image_file, # ‘./data/nii_10/case_20591/image.nii.gz’
    fat_file, # ‘./data/nii_10/case_20591/seg_at_mask.nii.gz’
    None,
    lung_file # ‘./data/nii_10/case_20591/lobe.nii.gz’
)
save_nii(ct_at_left_patch, './data/nii_test/case_20591/ct_at_left_patch.nii.gz')
save_nii(ct_at_right_patch, './data/nii_test/case_20591/ct_at_right_patch.nii.gz')
```
##### Parameter Description
- image_file: Path to the original CT image (image.nii.gz).
- fat_file: Path to the adipose tissue mask (seg_at_mask.nii.gz).
- lung_file: Path to the lung mask (lobe.nii.gz). Manual annotation, or Automatic prediction using an adipose tissue segmentation model.

##### Generated Output

After patch extraction, the generated bilateral adipose patches can be saved using the save_nii() function in the following format:
``` bash
case_xxxx/
├── ct_at_left_patch.nii.gz
└── ct_at_right_patch.nii.gz
```
These files can then be directly used as input for BATNet classification.
#### Important Note
All mask files must be stored as binary masks:
- 1 indicates foreground.
- 0 indicates background.
This applies to: lobe.nii.gz and seg_at_mask.nii.gz

## Training
BATNet consists of two complementary components:
- MG-ATSeg: Multi-granularity adipose tissue segmentation.
- CE-BATCls: Cross-enhanced brown adipose tissue classification.
Both models are implemented in PyTorch and trained using a carefully designed multi-stage optimization strategy. To facilitate reproducibility, we also provide a small demonstration dataset that allows users to quickly verify the complete training pipeline.

### Sample Training Data

To help users quickly verify the training pipeline, we provide a lightweight demonstration dataset.
- 3 cases for MG-ATSeg training
- 5 cases for CE-BATCls training

The sample dataset can be downloaded from Zenodo: https://zenodo.org/records/15524145/files/data.zip?download=1.
After downloading, extract the archive into the BATNet root directory.
#### Directory Structure
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
#### MG-ATSeg Dataset Structure
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

#### CE-BATCls Dataset Structure
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
##### CE-BATCls Data Preparation
BATNet supports two flexible data preparation strategies for CE-BATCls training.
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
- fat_mask.nii.gz: Adipose tissue mask. If this file is unavailable, BATNet will automatically invoke the MG-ATSeg model to generate it during training.
- brown_fat_mask.nii.gz: Ground-truth brown adipose tissue annotation. This file is optional but strongly recommended for supervised training.
- lobe.nii.gz: Lung mask.
During training, BATNet automatically performs:
- Bilateral adipose patch extraction
- Label generation
- End-to-end CE-BATCls optimization
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
import nibabel as nib
from pro_at_patch import SymmetricalATProcessor, save_nii

# Initialize processor
processor = SymmetricalATProcessor(
    model_file="./model_weights/mg_atseg_model.pth"
)

# Generate bilateral adipose patches
ct_at_left_patch, ct_at_right_patch, \
ct_at_left_label, ct_at_right_label = processor.process(
    image_file,   # ./data/cls_data/case_001/train/image.nii.gz
    fat_file,     # ./data/cls_data/case_001/train/fat_mask.nii.gz
    bat_file,     # ./data/cls_data/case_001/train/brown_fat_mask.nii.gz
    lung_file     # ./data/cls_data/case_001/train/lobe.nii.gz, Can None
)

# Save outputs
nib.save(nib.Nifti1Image(ct_at_left_patch, np.eye(4)), './data/cls_data/case_001/ct_at_left_patch.nii.gz')
nib.save(nib.Nifti1Image(ct_at_right_patch, np.eye(4)),  './data/cls_data/case_001/ct_at_right_patch.nii.gz')
nib.save(nib.Nifti1Image(ct_at_left_label.astype(np.uint8), np.eye(4)), './data/cls_data/case_001/ct_at_left_label.nii.gz')
nib.save(nib.Nifti1Image(ct_at_right_label.astype(np.uint8), np.eye(4)), './data/cls_data/case_001/ct_at_right_label.nii.gz')

```
###### Generated Files
```bash
case_xxxx/
├── ct_at_left_patch.nii.gz
├── ct_at_right_patch.nii.gz
├── ct_at_left_label.nii.gz
└── ct_at_right_label.nii.gz
```
These precomputed files can then be directly used by CE-BATCls without requiring online patch extraction.
##### Recommendation
- For quick experiments or small datasets, use Option 1.
- For large-scale training or repeated experiments, use Option 2 to improve efficiency and reduce preprocessing overhead.

### Training Commands
#### MG-ATSeg Training
```bash
cd BATNet
python MG_ATSeg/train.py
```

#### CE-BATCls Training
For simplicity, we demonstrate the end-to-end training pipeline, where adipose segmentation, patch extraction, and BAT classification are performed automatically.
```bash
cd BATNet
python CE_BATCIs/ce_train.py
```
#### Notes
- MG-ATSeg must be trained before CE-BATCls if adipose masks are not already available.
- During CE-BATCls training, missing adipose masks and adipose patches will be automatically generated when necessary.
- For large-scale experiments, we recommend precomputing all patches offline using pro_at_patch.py to accelerate training.


## model-architecture
### MG-ATSeg Components

| Module | Description | Location |
|-----------|--------------|-------------|
| MAM |   Mirror Attention Module | MG_BATCIs/model/unet_mam.py |

### CE-BATCls Components

| Module | Description | Location |
|-----------|--------------|-------------|
| CIM |   Contralateral Interaction | CE-BATCls/model/resnet_cim.py |
| Patch Processing Module |   Symmetrical AT Patch Processing Module | pro_at_patch.py |
| Transformer |   Case-Level Prediction | CE-BATCls/model/resnet_cim.py |

#### Note! We recommend using Ubuntu 22.04 or later for optimal compatibility and performance.
   
