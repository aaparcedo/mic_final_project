import os
import torch
import numpy as np
from torch.utils.data import Dataset, random_split
import glob
from pathlib import Path
from torchvision.transforms import Compose, Resize, Normalize

class LIDCIDRI2DDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None, split=None, train_ratio=0.7, val_ratio=0.15, seed=42):
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
            clean_images = sorted(list(clean_image_dir.glob("*.npy")))
            
            for img_path in clean_images:
                # Extract clean file info
                img_name = img_path.name
                
                # Construct potential clean mask path (just to be safe)
                clean_mask_path = self.root_dir / "clean" / "masks" / img_name
                
                # Check if this clean image has a mask (unlikely based on description)
                has_mask = clean_mask_path.exists()
                
                self.samples.append({
                    "image_path": img_path,
                    "mask_path": clean_mask_path if has_mask else None,
                    "has_mask": has_mask,
                    "case_id": "clean",
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load the image
        image = np.load(sample_info["image_path"])
        image = torch.from_numpy(image).float()
        
        # Apply transforms to the image if specified
        if self.transform:
            image = self.transform(image)
        
        # If there's a mask, load it too
        if sample_info["has_mask"]:
            mask = np.load(sample_info["mask_path"])
            mask = torch.from_numpy(mask).float()
            
            # Apply transforms to the mask if specified
            if self.mask_transform:
                mask = self.mask_transform(mask)
        
        # print(f'image shape: {image.shape}')
        # print(f'mask shape: {mask.shape}')
        return {
            "image": image,
            "mask": mask,
            "has_mask": sample_info["has_mask"],
            "case_id": sample_info["case_id"],
            "img_name": sample_info["img_name"]}


def get_model_specific_transform(model_type):
    if model_type == "MobileSAM":
        return Compose([
            Resize((1024, 1024)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_type == "PMFSNet":
        return Compose([Resize((256, 256))])  # Match PMFSNet input size
    else:  # UNet
        return Compose([Resize((512, 512))])