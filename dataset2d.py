import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path

DATASET_DIR = '/home/cap5516.student2/LIDC-IDRI-2D'
AUGMENTED_DATASET_DIR = '/home/cap5516.student2/LIDC-IDRI-2D-Augmented'

class LIDCIDRI2DDataset(Dataset):
    def __init__(self, root_dir=DATASET_DIR, transform=None, mask_transform=None, split=None, train_ratio=0.7, val_ratio=0.15, seed=42):
        """
        LIDC-IDRI 2D Dataset loader
        
        Args:
            root_dir (str): Root directory of the LIDC-IDRI-2D dataset
            transform (callable, optional): Transform to be applied on images
            mask_transform (callable, optional): Transform to be applied on masks
            split (str, optional): Dataset split: 'train', 'val', 'test', or None for all data
            train_ratio (float): Proportion of data used for training
            val_ratio (float): Proportion of data used for validation
            seed (int): Random seed for reproducible splits
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.diagnosis_information = self.get_diagnosis_information()
        
        # Get all images from regular data
        self.image_dirs = sorted(list(Path(self.root_dir / "images").glob("LIDC-IDRI-*")))
        
        # Store all image paths with corresponding mask paths (if they exist)
        self.samples = []
        
        # Add all regular samples with or without masks
        for image_dir in self.image_dirs:
            image_files = sorted(list(image_dir.glob("*.npy")))
            
            for img_path in image_files:
                # Extract file name components
                case_id = image_dir.name  # e.g., LIDC-IDRI-0068
                img_name = img_path.name  # e.g., 0068_NI000_slice117.npy
                
                if 'z_norm' in img_name or 'bias_field' in img_name or 'noise' in img_name:
                        continue
                
                # Construct potential mask path
                mask_path = self.root_dir / "masks" / case_id / img_name.replace('NI', 'MA')
                
                # Check if this image has a mask
                has_mask = mask_path.exists()
                
                # Add all samples
                self.samples.append({
                    "image_path": img_path,
                    "mask_path": mask_path if has_mask else None,
                    "has_mask": has_mask,
                    "case_id": case_id,
                    "img_name": img_name
                })
        
        # Add "clean" samples
        clean_image_dir = self.root_dir / "clean" / "images"
        if clean_image_dir.exists():
            clean_dirs = sorted(list(clean_image_dir.glob("LIDC-IDRI-*")))
            
            for clean_dir in clean_dirs:
                clean_images = sorted(list(clean_dir.glob("*.npy")))
                
                for img_path in clean_images:
                    # Extract clean file info
                    img_name = img_path.name
                    
                    if 'z_norm' in img_name or 'bias_field' in img_name or 'noise' in img_name:
                        continue
                    
                    # Construct potential clean mask path
                    clean_mask_path = self.root_dir / "clean" / "masks" / clean_dir.name / img_name.replace('CN', 'CM')
                    
                    # Check if this clean image has a mask
                    has_mask = clean_mask_path.exists()
                    
                    self.samples.append({
                        "image_path": img_path,
                        "mask_path": clean_mask_path if has_mask else None,
                        "has_mask": has_mask,
                        "case_id": clean_dir.name,
                        "img_name": img_name
                    })
        
        # Create data splits if requested
        if split is not None:
            # Set random seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Determine split sizes
            total_size = len(self.samples)
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size
            
            # Create a copy of samples for shuffling
            all_samples = self.samples.copy()
            np.random.shuffle(all_samples)
            
            # Split the dataset
            if split == 'train':
                self.samples = all_samples[:train_size]
            elif split == 'val':
                self.samples = all_samples[train_size:train_size + val_size]
            elif split == 'test':
                self.samples = all_samples[train_size + val_size:]
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
    
    def get_diagnosis_information(self):
        diagnosis_information = {}
        try:
            with open('patient_ids_with_diagnosis.txt', 'r') as file:
                for entry in file:
                    case_id = entry.strip()[:-2]
                    diagnosis = entry.strip()[-1]
                    diagnosis_information[case_id] = diagnosis
        except FileNotFoundError:
            print("Warning: patient_ids_with_diagnosis.txt not found. Using default diagnosis value.")
        return diagnosis_information
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Get diagnosis if available, otherwise use default value (0)
        diagnosis = torch.tensor(int(self.diagnosis_information.get(sample_info['case_id'], 0)))
        
        # Load the image
        image = np.load(sample_info["image_path"])
        image = torch.from_numpy(image).float()
        
        # Apply transforms to the image if specified
        if self.transform:
            image = self.transform(image)
        
        # Default mask
        mask = torch.zeros_like(image)
        
        # If there's a mask, load it too
        if sample_info["has_mask"] and sample_info["mask_path"]:
            try:
                mask = np.load(sample_info["mask_path"])
                mask = torch.from_numpy(mask).float()
                
                # Apply transforms to the mask if specified
                if self.mask_transform:
                    mask = self.mask_transform(mask)
            except Exception as e:
                print(f"Error loading mask: {sample_info['mask_path']}. Error: {e}")
                # Keep default zero mask
        
        # Ensure image has 3 dimensions if it's 2D
        if image.dim() == 2:
            image = image.unsqueeze(0)  # Add channel dimension
        
        # Ensure mask has same dimension as image
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension
            
        return {
            "image": image,
            "mask": mask,
            "diagnosis": diagnosis,
            "has_mask": sample_info["has_mask"],
            "case_id": sample_info["case_id"],
            "img_name": sample_info["img_name"],
            "image_path": str(sample_info['image_path'])
        }


class LIDCIDRIAugmentedDataset(Dataset):
    """
    Extended dataset class that combines original and augmented data
    """
    def __init__(self, original_root_dir=DATASET_DIR, augmented_root_dir=AUGMENTED_DATASET_DIR, transform=None, mask_transform=None, 
                split=None, train_ratio=0.7, val_ratio=0.15, seed=42, use_augmented=True):
        
        # Load the original dataset
        self.original_dataset = LIDCIDRI2DDataset(
            root_dir=original_root_dir,
            transform=transform,
            mask_transform=mask_transform,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed
        )
        
        # Only load augmented data if requested
        if use_augmented:
            # Load the augmented dataset
            self.augmented_dataset = LIDCIDRI2DDataset(
                root_dir=augmented_root_dir,
                transform=transform,
                mask_transform=mask_transform,
                split=split,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed
            )
            
            # Combine datasets
            self.combined_dataset = ConcatDataset([self.original_dataset, self.augmented_dataset])
        else:
            # Just use the original dataset
            self.combined_dataset = self.original_dataset
    
    def __len__(self):
        return len(self.combined_dataset)
    
    def __getitem__(self, idx):
        return self.combined_dataset[idx]


# def get_model_specific_transform(model_type):
#     if model_type == "MobileSAM":
#         return Compose([
#             Resize((1024, 1024)),
#             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#     elif model_type == "PMFSNet":
#         return Compose([Resize((256, 256))])  # Match PMFSNet input size
#     else:  # UNet
#         return Compose([Resize((512, 512))])
