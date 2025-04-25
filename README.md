## Description
Lung Nodule Localization and Classification in CT Images

## Installation
1. Clone the repository (and necessary repositories):
```bash
git clone https://github.com/aaparcedo/mic_final_project.git
cd mic_final_project
git clone https://github.com/ChaoningZhang/MobileSAM.git
```

2. Create and activate a virtual environment:
```bash
conda create -n nodule python=3.10 -y
conda activate nodule
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset
Dataset has to be downloaded from the [Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/lidc-idri/) with the [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images).
Once you have the dataset, update the paths in the make2ddataset.py file.
Since we're doing localization and diagnosis prediction we only use the samples that have a final diagnosis, these are in the file paths_with_diagnosis.txt.
```bash
python make2ddataset.py
```

## Experiments
The training script is setup to run UNet, PMFSNet, and MobileSAM. Update the configuration inside the file with the same of the model you want to run.
```bash
python train.py
```
