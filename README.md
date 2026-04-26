# BATNet: Symmetry-Aware Deep Learning for Brown Adipose Tissue Detection

![BATNet Workflow]() For details of the model structure, please refer to the paper

## Table of Contents
1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Data Preparation](#data-preparation) 
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Sample Data](#Sample-Data)
8. [Inference validation](#Inference-validation)
9. [Docker Environment for Quick Validation](#Docker-Environment-for-Quic-Validation)

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

### Prerequisites
- Ubuntu 24.04.2 LTS
- Python 3.8.2
- Pytorch 2.4.1+cu121
This repository has been tested on Tesla V100-SXM2-32GB. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

### Setup
```bash
# Clone repository
git clone --recurse-submodules https://github.com/zhjtwx/BATNet.git
cd BATNet
# Install core packages
1.Create Conda Environment
conda create -n batnet python=3.8 -y
conda activate batnet

2.Install PyTorch with CUDA 12.4
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

3.Install Other
pip install -r requirements.txt
```
### Expected Install Time
The installation typically takes:
- **15-25 minutes** on a normal desktop with good internet connection
- May extend to **30-45 minutes** if compilation is required for your specific system

## Quick Start

### Inference Example

```
python bat_inf.py --input data/crop_744/info.csv --output crop_744_inf.csv
python bat_inf.py --input data/nii_10 --output nii_10_inf.csv
```
    --input
    Supported input formats:

    1. CSV file
       - Must contain a 'case_id' column, The case_id stores the input path for each case.

    2. Single root directory
       - Recursively scans all valid case folders.

    3. Multiple case directories
       - Filters and keeps only valid cases.

    A valid case directory must satisfy one of the following:

    A. Contains both:
       - image.nii.gz # Original image
       - lobe.nii.gz # lung mask segmented from the original image.

    B. Contains both:
       - ct_at_left_patch.nii.gz # Patches on the left side, cropped from the original image based on the at region.
       - ct_at_right_patch.nii.gz # Patches on the right side, cropped from the original image based on the at region.
 

    --output: CSV file for saving results
      output result: The contents of the CSV file are as follows.

| case_id | prediction |
|-----------|--------------|
| input case id | Predicted probability of BAT|

    If the input case is as follows, both the fat segmentation model and the brown fat classification model will be run simultaneously. In addition to outputting the above table, the segmented fat will also be saved in the same path as the original image.
       - image.nii.gz # Original image
       - lobe.nii.gz # lung mask segmented from the original image.

    If the input case is as follows, only the brown fat classification model will be run.
       - ct_at_left_patch.nii.gz # Patches on the left side, cropped from the original image based on the at region.
       - ct_at_right_patch.nii.gz # Patches on the right side, cropped from the original image based on the at region.


## Data Preparation
### Directory Structure
```python
data/
├──seg_data/  # Data for MG_ATSeg    
│    ├── train/
│    │   ├── case_001/
│    │   │   ├── data_png/
│    │   │   │      ├── 0.png
│    │   │   │      ├── 1.png
│    │   │   │      └── ...
│    │   │   └── data_mask/
│    │   │          ├── 0.png
│    │   │          ├── 1.png
│    │   │          └── ...
│    │   └── case_002/
│    └── test/
└──cls_data/ # Data for CE_BATCIs
     ├── train/
     │   ├── case_001/
     │   │   ├── image.nii.gz  # The entire CT is saved in nii format. This path must exist. Input for MG_ATSeg model inference. Each volume has dimensions of (Z, X, Y), aligned along the Z-axis from top to bottom.
     │   │   ├── fat_mask.nii.gz  # Fat mask. If this path is missing, the MG_ATSeg fat segmentation model will be automatically called for fat segmentation.
     │   │   ├── brown_fat_mask.nii.gz # Brown fat mask, manually marked with software, if empty, the default brown fat mask mark value is all 0
     │   │   ├── lobe.nii.gz # This path contains the lung segmentation mask corresponding to the CT scan, the purpose of which is to extract fat above the armpit and remove fat below the armpit.
     │   │   ├── ct_at_left_label.nii.gz # Labels for left patches, You can call "pro_at_patch.py" to generate it offline, or you can generate it directly in the training model.
     │   │   ├── ct_at_right_label.nii.gz # Labels for right patches, You can call "pro_at_patch.py" to generate it offline, or you can generate it directly in the training model.
     │   │   ├── ct_at_left_patch.nii.gz # left patches, You can call "pro_at_patch.py" to generate it offline, or you can generate it directly in the training model.
     │   │   └── ct_at_right_patch.nii.gz # right patches, You can call "pro_at_patch.py" to generate it offline, or you can generate it directly in the training model.
     │   └── case_002/
     └── test/
```
### Preprocessing Script
```python
import sys
sys.path.append('.')
from pro_at_patch import SymmetricalATProcessor
# Initialize 
processor = SymmetricalATProcessor(
    model_file="mg_stseg.pth"  #seg_model_file,Can be None, but when None, there must be a fat_mask.nii.gz file
)
# Run 
ct_at_left, ct_at_right, left_label, right_label = processor.process(image_file, fat_file, bat_file, lung_file)
"""
image_file: Path of the original image (in nii.gz format)；
fat_file: Path of the fat mask (in nii.gz format), can be None. When it is None, the fat segmentation model (MG_BATCIs) will be invoked to obtain the fat mask.
bat_file: Path of the bat mask (in nii.gz format), derived from manually annotated brown adipose tissue.
lung_file: Path of the lung mask (in nii.gz format), Lung mask segmentation derived from TotalSegmentator.
"""
```
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

## Training
### MG-ATSeg Training
```bash
cd BATNet
python MG_ATSeg/train.py # Please modify the corresponding hyperparameters in train.py
```
### CE-BATCls Training
```bash
cd BATNet
python CE_BATCIs/ce_train.py # Please modify the corresponding hyperparameters in ce_train.py
```

## Sample Data

Verification of the training process: We have uploaded 8 chest CT images from the dataset as a demonstration for training (3 for fat segmentation and 5 for brown fat classification). Please download the data from the following link: https://zenodo.org/records/15524145/files/data.zip?download=1. After downloading, please unzip the data into the BATNet directory. The directory structure of the data can be found in the "Directory Structure" section. Once downloaded, simply run "MG-ATSeg Training" and "CE-BATCls Training" to train the respective models.

## Inference validation
### Model Weights 
The model weights can be downloaded from Zenodo: 
https://zenodo.org/records/17540420/files/model_weights.zip?download=1; Password: batnet
Note: After downloading, please extract the contents to the root directory of the BATNet project.
Password: batnet

### Validation dataset
To support comprehensive validation, we provided adipose tissue segmentation patches extracted from the holdout set (n = 744), which were used for evaluating the classification module. We also provided 10 complete CT scans (.nii format) selected from the holdout set, enabling end-to-end validation of the entire pipeline, including both adipose segmentation and whole-slice BAT classification.

The adipose tissue segmentation patches (n = 744) and complete CT scans (n = 10) can be downloaded from Zenodo:
https://zenodo.org/records/17541503/files/data.zip?download=1; Password: batnet

### infer test
Once both the model and data are in place, users can run the inference script directly via:
1. python CE_BATCIs/ce_infer.py
2. Follow the inference procedure described in the Inference Example section.

## Docker Environment for Quick Validation
To facilitate quick validation and testing, we provide a pre-configured Docker environment. This package contains a fully set-up runtime environment, along with code, datasets, and pre-trained model weights. 
### Download Link
[https://zenodo.org/api/records/19756491/files-archive] 
### Quick Start
```bash
docker load -i batnet-image-v2.tar.gz # Load the Docker image after downloading
docker run --gpus all -it --rm \
  --runtime=nvidia \
  --privileged \
  --security-opt seccomp=unconfined \
  --shm-size=16g \
  -e OPENBLAS_NUM_THREADS=1 \
  batnet-image:v2 /bin/bash  # Verify successful image import and launch the container
```
#### Note! We recommend using Ubuntu 22.04 or later for optimal compatibility and performance.
   
