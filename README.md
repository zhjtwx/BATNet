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
8. [Reproduction Instructions](#Reproduction-Instructions)

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
- Ubuntu 16.04.4 LTS
- Python 3.6.13
- Pytorch 1.10.0+cu113
- NVIDIA GPU + CUDA_10.1 CuDNN_8.2
This repository has been tested on NVIDIA TITANXP. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

### Setup
```bash
# Clone repository
git clone --recurse-submodules https://github.com/zhjtwx/BATNet.git
cd BATNet
# Install core packages
1.Create Conda Environment
conda create -n batnet python=3.6.13 -y
conda activate batnet
2.Upgrade Pip and Install System Dependencies
pip install --upgrade pip
apt update && apt install -y libxml2 libgl1-mesa-glx  # Essential libraries
3.Install CUDA Toolkit
conda install -c conda-forge cudatoolkit=11.3 -y  # Use conda-forge for better compatibility
4.Install PyTorch with CUDA 11.3
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
5.Install Other
pip install -r requirements.txt
```
### Expected Install Time
The installation typically takes:
- **15-25 minutes** on a normal desktop with good internet connection
- May extend to **30-45 minutes** if compilation is required for your specific system

### Notes
Some packages like pycuda and mmcv-full require compatible system environments. GPU-related packages will auto-detect CUDA 10.1 by default. If interrupted, you can resume installation with --ignore-installed flag

## Quick Start

### Inference Example

#### At segmentation model inference example for the whole case
```python
import sys
sys.path.append('./MG_ATSeg')
from patient_fat_inf import InfATMask
# Initialize 
at_seg_model = InfATMask(model_file="mg_stseg.pth", # MG_ATSEG training saved model
                            device_ids=[1,2] 
                            )
# Run seg
# The input is the entire DICOM saved nii format data，The output is the fat segmentation mask corresponding to the entire nii
at_seg_mask = seg_at_model.inf_case_at(ct_nii_path='./data/cls_data/train/case_001/image.nii.gz') 
```

#### Classification of BAT
```python
import sys
sys.path.append('./CE_BATCIs')
from ce_infer import InfATMask
# Initialize 
predictor = BATInference(
        model_path='ce_batcis.pth', # CE_BATCIs training saved model
        device='cuda'  # or 'cpu'
    )
# Run classification
out_pres = predictor.batch_predict(
            data_dir='./data/nii_10/case_*'
          )  

# Batch input is supported. See the "Data Preparation" section for input format details. The output shows the predicted probability of brown adipose tissue for each case.
```


#### Model inference time
On an NVIDIA TITAN XP GPU, BATNet processes standard chest CT scans (512×512×300 slices) in 15-30 seconds per case end-to-end, with peak VRAM usage of 10.5GB. The pipeline comprises: (1) 10-second preprocessing (HU normalization/resampling), (2) 3-6 seconds/slice adipose tissue segmentation using the Mirror Attention U-Net, and (3) 2-5 seconds contralateral patch analysis via the ResNet-Transformer classifier. Continuous processing achieves 2-3 cases/minute throughput. Demo executions on the provided 8-case sample dataset complete in 2-4 minutes, with variations depending on slice count (see Sample Data). 

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
ct_at_left, ct_at_right, left_label, right_label = processor.process(image_file, fat_file, bat_file)
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
python CE-BATCls/ce_train.py # Please modify the corresponding hyperparameters in ce_train.py
```

## Sample Data
We uploaded 8 chest CT images from the dataset for demonstration (3 for fat segmentation and 5 for brown fat classification). Please download the data at the following link (https://zenodo.org/records/15524145/files/data.zip?download=1). After downloading the data, please unzip the data under BATNet. Its distribution can be viewed in the directory structure section. Please note that the sample data is provided only to allow users to verify the workflow of the provided code. Since model weights are a key component of the BATNet model, which needs to be applied to business in the future, we cannot disclose the specific values ​​of model weights at present. Users can train the model with their own datasets to obtain their own model weights.

## Reproduction Instructions
To demonstrate the workflow using the provided 8 sample cases download the data from Zenodo and run the demo, which will complete in approximately 10-20 minutes on an NVIDIA TITAN XP GPU. Please note this simplified demo uses processed intermediate outputs to verify the pipeline architecture, while the full research implementation requires training with larger datasets. Due to ongoing clinical deployment preparations, the complete model weights and full training data are not currently available for public release. Researchers are encouraged to train the model using their own datasets following our architecture specifications, and may contact the authors for potential collaboration opportunities to access extended validation datasets.
