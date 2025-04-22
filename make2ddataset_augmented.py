import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import albumentations as A
from scipy.ndimage import gaussian_filter

# Original directories
IMAGE_DIR = Path('/home/cap5516.student2/LIDC-IDRI-2D/images')
MASK_DIR = Path('/home/cap5516.student2/LIDC-IDRI-2D/masks')
CLEAN_IMAGE_DIR = Path('/home/cap5516.student2/LIDC-IDRI-2D/clean/images')
CLEAN_MASK_DIR = Path('/home/cap5516.student2/LIDC-IDRI-2D/clean/masks')

# Augmented directories - using same structure as original for dataset compatibility
# We'll keep the directory structure the same, but in a new "augmented" parent directory
AUG_ROOT_DIR = Path('/home/cap5516.student2/LIDC-IDRI-2D-Augmented')
AUG_IMAGE_DIR = AUG_ROOT_DIR / 'images'
AUG_MASK_DIR = AUG_ROOT_DIR / 'masks'
AUG_CLEAN_IMAGE_DIR = AUG_ROOT_DIR / 'clean/images'
AUG_CLEAN_MASK_DIR = AUG_ROOT_DIR / 'clean/masks'

# Create augmentation directories
for dir_path in [AUG_IMAGE_DIR, AUG_MASK_DIR, AUG_CLEAN_IMAGE_DIR, AUG_CLEAN_MASK_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Define augmentation functions
def create_augmentation_pipeline():
    """Create a set of augmentation pipelines using albumentations"""
    
    # Pipeline 1: Flip augmentations
    flip_aug = A.Compose([
        A.HorizontalFlip(p=1.0),
    ])
    
    # Pipeline 2: Vertical flip
    vflip_aug = A.Compose([
        A.VerticalFlip(p=1.0),
    ])
    
    # Pipeline 3: Affine transformations
    affine_aug = A.Compose([
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), 
                rotate=(-15, 15), shear=(-10, 10), p=1.0),
    ])
    
    # Pipeline 4: Noise augmentation (fixed parameter name)
    noise_aug = A.Compose([
        A.GaussNoise(var_limit=(10, 50), mean=0, p=1.0),
    ])
    
    # Return a dictionary of augmentation pipelines
    return {
        "flip": flip_aug,
        "vflip": vflip_aug,
        "affine": affine_aug,
        "noise": noise_aug
    }

def apply_blur(image, sigma=1.0):
    """Apply Gaussian blur to the image"""
    return gaussian_filter(image, sigma=sigma)

def apply_bias_field(image, bias_magnitude=0.2):
    """Apply a simulated bias field to the image"""
    # Create a simple bias field (could be more sophisticated)
    y, x = np.indices(image.shape)
    y_center = image.shape[0] // 2
    x_center = image.shape[1] // 2
    
    # Create a radial gradient from the center
    bias = np.sqrt(((y - y_center) / image.shape[0]) ** 2 + 
                  ((x - x_center) / image.shape[1]) ** 2)
    
    # Scale and apply the bias
    bias = 1.0 + bias_magnitude * bias
    return image * bias

def apply_z_normalization(image):
    """Apply Z-score normalization to the image"""
    mean = np.mean(image)
    std = np.std(image)
    if std == 0:
        return image - mean
    return (image - mean) / std

def apply_augmentations(image, mask, aug_pipeline, aug_name):
    """Apply an augmentation pipeline to image and mask"""
    # Convert mask to the right format for albumentations (it expects uint8)
    # This fixes the "src data type = bool is not supported" error
    if mask.dtype == bool:
        mask = mask.astype(np.uint8)
    
    # Make sure image is in float32 for augmentations
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    if aug_name in ["flip", "vflip", "affine"]:
        # For spatial augmentations, apply to both image and mask
        augmented = aug_pipeline(image=image, mask=mask)
        return augmented['image'], augmented['mask']
    
    elif aug_name == "noise":
        # For noise, only apply to image
        augmented = aug_pipeline(image=image)
        return augmented['image'], mask
    
    elif aug_name == "blur":
        # Apply blur only to image
        return apply_blur(image), mask
    
    elif aug_name == "bias_field":
        # Apply bias field only to image
        return apply_bias_field(image), mask
    
    elif aug_name == "z_norm":
        # Apply Z-normalization only to image
        return apply_z_normalization(image), mask
    
    return image, mask

def save_augmented_data(original_image, original_mask, patient_id, nodule_idx, slice_idx, 
                        is_clean=False, prefix_list=None):
    """Apply all augmentations and save the results in a format compatible with the dataset class"""
    
    # Get the augmentation pipelines
    aug_pipelines = create_augmentation_pipeline()
    
    # Determine appropriate directories based on clean status
    if is_clean:
        base_image_dir = AUG_CLEAN_IMAGE_DIR / patient_id
        base_mask_dir = AUG_CLEAN_MASK_DIR / patient_id
        name_prefix = "CN"
        mask_prefix = "CM"
    else:
        base_image_dir = AUG_IMAGE_DIR / patient_id
        base_mask_dir = AUG_MASK_DIR / patient_id
        name_prefix = "NI"
        mask_prefix = "MA"
    
    # Create directories if they don't exist
    base_image_dir.mkdir(parents=True, exist_ok=True)
    base_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply and save each augmentation
    augmentations = list(aug_pipelines.items()) + [
        ("blur", None), 
        ("bias_field", None), 
        ("z_norm", None)
    ]
    
    for aug_name, aug_pipeline in augmentations:
        try:
            # Apply the augmentation
            aug_image, aug_mask = original_image.copy(), original_mask.copy()
            
            # Apply the augmentation
            if aug_pipeline is not None:
                aug_image, aug_mask = apply_augmentations(aug_image, aug_mask, aug_pipeline, aug_name)
            else:
                # Apply non-albumentations augmentations
                aug_image, aug_mask = apply_augmentations(aug_image, aug_mask, None, aug_name)
            
            # Use the original naming pattern to ensure compatibility with dataset class
            # Just add the augmentation type to maintain the same file structure
            # The dataset class expects: XXXX_NIYYY_sliceZZZ.npy and XXXX_MAYYY_sliceZZZ.npy
            nodule_name = f"{patient_id[-4:]}_{name_prefix}{prefix_list[nodule_idx]}_slice{prefix_list[slice_idx]}"
            mask_name = f"{patient_id[-4:]}_{mask_prefix}{prefix_list[nodule_idx]}_slice{prefix_list[slice_idx]}"
            
            # Save the augmented data with the augmentation type appended
            np.save(base_image_dir / f"{nodule_name}_{aug_name}.npy", aug_image)
            np.save(base_mask_dir / f"{mask_name}_{aug_name}.npy", aug_mask)
            
            print(f"Saved augmented {aug_name} image and mask for {nodule_name}")
        except Exception as e:
            print(f"Error applying {aug_name} augmentation: {str(e)}")
            continue

# Main code for augmentation only (not generating the original dataset again)
def augment_existing_dataset():
    """
    Augment the existing dataset without recreating the original files.
    This function loads the already generated images and masks and creates augmented versions.
    """
    print("Starting augmentation of existing LIDC-IDRI-2D dataset...")
    prefix = [str(x).zfill(3) for x in range(1000)]
    
    # Process regular images with nodules
    patient_dirs = sorted(list(Path(IMAGE_DIR).glob("LIDC-IDRI-*")))
    print(f"Found {len(patient_dirs)} patient directories")
    
    for patient_dir in tqdm(patient_dirs):
        patient_id = patient_dir.name  # e.g., LIDC-IDRI-0068
        aug_patient_image_dir = AUG_IMAGE_DIR / patient_id
        aug_patient_mask_dir = AUG_MASK_DIR / patient_id
        
        # Create patient-specific directories in augmented location
        aug_patient_image_dir.mkdir(parents=True, exist_ok=True)
        aug_patient_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images for this patient
        image_files = sorted(list(patient_dir.glob("*.npy")))
        print(f"Processing {len(image_files)} images for patient {patient_id}")
        
        for image_path in image_files:
            # Extract information from filename
            img_name = image_path.stem  # e.g., 0068_NI000_slice117
            
            # Parse nodule index and slice index from filename
            parts = img_name.split('_')
            case_suffix = parts[0]  # e.g., 0068
            nodule_type = parts[1][:2]  # e.g., NI
            nodule_idx = int(parts[1][2:])  # e.g., 000
            slice_idx = int(parts[2].replace('slice', ''))  # e.g., 117
            
            # Construct matching mask path
            mask_type = "MA" if nodule_type == "NI" else "CM"
            mask_filename = f"{case_suffix}_{mask_type}{parts[1][2:]}_slice{parts[2].split('slice')[1]}.npy"
            mask_path = MASK_DIR / patient_id / mask_filename
            
            # Check if mask exists
            if not mask_path.exists():
                print(f"Warning: No mask found for {image_path}. Skipping.")
                continue
            
            # Load image and mask
            image = np.load(image_path)
            mask = np.load(mask_path)
            
            print(f"Augmenting {img_name}.npy - Image shape: {image.shape}, Mask shape: {mask.shape}")
            
            # Apply augmentations - keeping original filename pattern for compatibility
            save_augmented_data(image, mask, patient_id, nodule_idx, slice_idx, 
                              is_clean=False, prefix_list=prefix)
    
    # Process clean images
    clean_dirs = sorted(list(Path(CLEAN_IMAGE_DIR).glob("LIDC-IDRI-*")))
    print(f"Found {len(clean_dirs)} clean patient directories")
    
    for clean_dir in tqdm(clean_dirs):
        patient_id = clean_dir.name
        aug_clean_image_dir = AUG_CLEAN_IMAGE_DIR / patient_id
        aug_clean_mask_dir = AUG_CLEAN_MASK_DIR / patient_id
        
        # Create patient-specific directories in augmented location
        aug_clean_image_dir.mkdir(parents=True, exist_ok=True)
        aug_clean_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all clean images for this patient
        clean_image_files = sorted(list(clean_dir.glob("*.npy")))
        
        for image_path in clean_image_files:
            # Extract information from filename
            img_name = image_path.stem
            
            # Parse nodule index and slice index from filename
            parts = img_name.split('_')
            case_suffix = parts[0]
            nodule_type = parts[1][:2]  # e.g., CN
            nodule_idx = int(parts[1][2:])  # e.g., 001
            slice_idx = int(parts[2].replace('slice', ''))
            
            # Construct matching mask path
            mask_filename = f"{case_suffix}_CM{parts[1][2:]}_slice{parts[2].split('slice')[1]}.npy"
            mask_path = CLEAN_MASK_DIR / patient_id / mask_filename
            
            # Check if mask exists
            if not mask_path.exists():
                print(f"Warning: No clean mask found for {image_path}. Creating zero mask.")
                # Create zero mask as per original code
                image = np.load(image_path)
                mask = np.zeros_like(image)
            else:
                # Load image and mask
                image = np.load(image_path)
                mask = np.load(mask_path)
            
            print(f"Augmenting clean image {img_name}.npy")
            
            # Apply augmentations for clean images
            save_augmented_data(image, mask, patient_id, nodule_idx, slice_idx, 
                              is_clean=True, prefix_list=prefix)

# Execute augmentation on existing dataset
if __name__ == "__main__":
    augment_existing_dataset()