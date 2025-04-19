import pylidc as pl
import os
from tqdm import tqdm
from pathlib import Path
from pylidc.utils import consensus
import numpy as np

DATASET_DIR = Path('/home/cap5516.student2/LIDC-IDRI')
IMAGE_DIR = Path('/home/cap5516.student2/LIDC-IDRI-2D/images')
MASK_DIR = Path('/home/cap5516.student2/LIDC-IDRI-2D/masks')
CLEAN_IMAGE_DIR = Path('/home/cap5516.student2/LIDC-IDRI-2D/clean/images')
CLEAN_MASK_DIR = Path('/home/cap5516.student2/LIDC-IDRI-2D/clean/masks')

PAD = 512
CONFIDENCE = 0.5
MASK_THRESHOLD = 8

patient_list = os.listdir(DATASET_DIR) # len(patient_list) = 158

prefix = [str(x).zfill(3) for x in range(1000)]

for patient_id in tqdm(patient_list):
    pid = patient_id # e.g. LIDC-IDRI-0068
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    nodules_annotation = scan.cluster_annotations()
    vol = scan.to_volume()
    print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))
    
    patient_image_dir = IMAGE_DIR / pid
    patient_mask_dir = MASK_DIR / pid
    Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
    Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)

    if len(nodules_annotation) > 0:
        # Patients with nodules
        for nodule_idx, nodule in enumerate(nodules_annotation):
        # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
        # This current for loop iterates over total number of nodules in a single patient
            mask, cbbox, masks = consensus(nodule,CONFIDENCE,PAD)
            lung_np_array = vol[cbbox]


            for nodule_slice in range(mask.shape[2]):
                # This second for loop iterates over each single nodule.
                # There are some mask sizes that are too small. These may hinder training.
                if np.sum(mask[:,:,nodule_slice]) <= MASK_THRESHOLD:
                    continue
                
                lung_np_array[lung_np_array==-0] = 0

                # This itereates through the slices of a single nodule
                # Naming of each file: NI= Nodule Image, MA= Mask Original
                nodule_name = "{}_NI{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                mask_name = "{}_MA{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice]) # e.g., 0068_MA001_slice148.npyc
                
                # import code; code.interact(local=locals())
                
                save_image = lung_np_array[:,:,nodule_slice]
                save_mask = mask[:,:,nodule_slice]
                
                print(f'save image shape: {save_image.shape}')
                print(f'save mask shape: {save_mask.shape}')
                assert save_image.shape != 2, "Image must be 2D tensor"
                assert save_mask.shape != 2, "Mask must be 2D tensor"
                
                np.save(patient_image_dir / nodule_name, save_image)
                np.save(patient_mask_dir / mask_name,save_mask)
    else:
        print("Clean Dataset",pid)
        patient_clean_dir_image = CLEAN_IMAGE_DIR / pid
        patient_clean_dir_mask = CLEAN_MASK_DIR / pid
        Path(patient_clean_dir_image).mkdir(parents=True, exist_ok=True)
        Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)
        #There are patients that don't have nodule at all. Meaning, its a clean dataset. We need to use this for validation
        for slice in range(vol.shape[2]):
            if slice >50:
                break
            lung_np_array[lung_np_array==-0] = 0

            #CN= CleanNodule, CM = CleanMask
            nodule_name = "{}_CN001_slice{}".format(pid[-4:],prefix[slice])
            mask_name = "{}_CM001_slice{}".format(pid[-4:],prefix[slice])
            
            save_image = lung_np_array[:,:,nodule_slice]
            save_mask = np.zeros_like(save_image)
                
            print(f'(clean) save image shape: {save_image.shape}')
            print(f'(clean) save mask shape: {save_mask.shape}')
            
            assert save_image.shape != 2, "Image must be 2D tensor"
            assert save_mask.shape != 2, "Mask must be 2D tensor"
            
            np.save(patient_clean_dir_image / nodule_name, save_image)
            np.save(patient_clean_dir_mask / mask_name, save_mask)