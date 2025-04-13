import os
import torch
import numpy as np
import pydicom
from torch.utils.data import Dataset
import glob
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from scipy.ndimage import zoom

BASE_PATH = os.path.expanduser("~")


class LungNoduleDataset(Dataset):
    """
    PyTorch Dataset for loading lung CT volumes with segmentation masks and diagnosis labels.
    
    This dataset class:
    1. Reads paths and diagnosis from a text file
    2. Loads DICOM files from each folder to create 3D volumes
    3. Loads corresponding segmentation masks
    4. Provides the diagnosis as classification labels
    5. Resizes volumes to reduce memory usage
    """
    
    def __init__(self, paths_file, transform=None, target_size=(256, 256), resize_slices=False):
        """
        Initialize the dataset.
        
        Args:
            paths_file (str): Path to the text file containing folder paths and diagnoses
            transform (callable, optional): Optional transform to be applied on samples
            target_size (tuple): Target size for width and height (default: 256x256)
            resize_slices (bool): Whether to resize slice dimension (default: False)
        """
        self.transform = transform
        self.target_size = target_size
        self.resize_slices = resize_slices
        
        # Read paths and diagnoses from the file
        self.samples = []
        with open(paths_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                path = BASE_PATH + line[1:-2]
                diagnosis = int(line[-1])
                self.samples.append((path, diagnosis))
                    
        print(f"Loaded {len(self.samples)} samples from {paths_file}")
        print(f"Volumes will be resized to {target_size[0]}×{target_size[1]}")
        
        self.size = len(self.samples)
        # Validate a few samples to ensure paths are correct
        self._validate_samples(num_to_check=min(5, self.size))
        
                
    def _validate_samples(self, num_to_check=5):
        """Validate a subset of samples to ensure paths exist"""
        for i, (path, _) in enumerate(self.samples[:min(num_to_check, len(self.samples))]):
            if not os.path.exists(path):
                print(f"Warning: Path does not exist: {path}")
            else:
                # Check if the path contains DICOM files
                dicom_files = self._get_dicom_files(path)
                print(f"Sample {i+1}: Path {path} - {len(dicom_files)} DICOM files")
                
                # Check if the path contains an XML file
                xml_files = glob.glob(os.path.join(path, "*.xml"))
                if xml_files:
                    print(f"  Found {len(xml_files)} XML files in the directory")
                else:
                    print(f"  Warning: No XML files found in the directory")
                
                # Get and print the shape of the volume
                if dicom_files:
                    shape = self._get_volume_shape(dicom_files)
                    print(f"  Volume shape: {shape[0]} slices × {shape[1]} height × {shape[2]} width")
                    print(f"  Will be resized to: {shape[0] if not self.resize_slices else shape[0]//2} slices × {self.target_size[0]} height × {self.target_size[1]} width")
    
    def _get_dicom_files(self, path):
        """Get sorted list of DICOM files in a folder"""
        # Search for DICOM files with common extensions
        dicom_files = []
        for ext in ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']:
            dicom_files.extend(glob.glob(os.path.join(path, ext)))
        
        # If no files found with extensions, try finding all files (some DICOM files have no extension)
        if not dicom_files:
            all_files = glob.glob(os.path.join(path, '*'))
            dicom_files = [f for f in all_files if os.path.isfile(f)]
        
        # Sort the files to ensure proper ordering
        dicom_files.sort()
        return dicom_files
    
    def _get_volume_shape(self, dicom_files):
        """Get the shape of the volume from DICOM files"""
        # Read the first DICOM to get height and width
        first_slice = pydicom.dcmread(dicom_files[0])
        height, width = first_slice.pixel_array.shape
        
        # Return the 3D shape (slices × height × width)
        return (len(dicom_files), height, width)
    
    def _resize_volume(self, volume):
        """Resize a 3D volume to target dimensions"""
        original_shape = volume.shape
        
        # Determine target depth (number of slices)
        if self.resize_slices:
            target_depth = original_shape[0] // 2  # Reduce slices by half
        else:
            target_depth = original_shape[0]  # Keep original slice count
        
        # Use scipy.ndimage.zoom for efficient 3D resizing
        # Compute scaling factors for each dimension
        scale_factors = (
            target_depth / original_shape[0],
            self.target_size[0] / original_shape[1],
            self.target_size[1] / original_shape[2]
        )
        
        # Apply zoom (interpolate) to resize the volume
        resized_volume = zoom(volume, scale_factors, order=1, mode='nearest')
        
        return resized_volume
        
    def _load_volume(self, path):
        """Load a 3D volume from DICOM files and resize it"""
        dicom_files = self._get_dicom_files(path)
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {path}")
        
        # Get shape information
        shape = self._get_volume_shape(dicom_files)
        
        # Initialize the volume array
        volume = np.zeros(shape, dtype=np.float32)
        
        # Load each slice
        for i, dcm_path in enumerate(dicom_files):
            ds = pydicom.dcmread(dcm_path)
            volume[i, :, :] = ds.pixel_array.astype(np.float32)
        
        # Normalize the volume to [0, 1] range
        volume = self._normalize_volume(volume)
        
        # Resize the volume
        volume = self._resize_volume(volume)
        
        return volume
    
    def _normalize_volume(self, volume):
        """Normalize the volume to [0, 1] range"""
        min_val = volume.min()
        max_val = volume.max()
        
        if max_val > min_val:
            volume = (volume - min_val) / (max_val - min_val)
        
        return volume
    
    def _resize_mask(self, mask, original_shape):
        """Resize a segmentation mask using the same approach as volume"""
        # Determine target depth (number of slices)
        if self.resize_slices:
            target_depth = original_shape[0] // 2  # Reduce slices by half
        else:
            target_depth = original_shape[0]  # Keep original slice count
        
        # Calculate scaling factors
        scale_factors = (
            target_depth / original_shape[0],
            self.target_size[0] / original_shape[1],
            self.target_size[1] / original_shape[2]
        )
        
        # Use order=0 to preserve binary nature of mask (nearest neighbor)
        resized_mask = zoom(mask, scale_factors, order=0, mode='nearest', prefilter=False)
        
        # Ensure mask remains binary
        resized_mask = (resized_mask > 0.5).astype(np.float32)
        
        return resized_mask
        
    def _load_mask(self, path):
        """
        Load segmentation mask for the volume from XML annotations and resize it.
        """
        # Initialize mask volume
        dicom_files = self._get_dicom_files(path)
        original_shape = self._get_volume_shape(dicom_files)
        mask_volume = np.zeros(original_shape, dtype=np.float32)
        
        # Find XML file in the directory
        xml_files = glob.glob(os.path.join(path, "*.xml"))
        if not xml_files:
            print(f"Warning: No XML file found in {path}. Creating zero mask.")
            # Resize the zero mask
            mask_volume = self._resize_mask(mask_volume, original_shape)
            return mask_volume
        
        # Parse the XML file and create mask
        try:
            tree = ET.parse(xml_files[0])
            root = tree.getroot()
            
            # Define namespace
            ns = {'ns': 'http://www.nih.gov'}
            
            # Find all nodules
            nodules = root.findall('.//ns:unblindedReadNodule', ns)
            if not nodules:
                # Try without namespace if no nodules found
                nodules = root.findall('.//unblindedReadNodule')
            
            # Dictionary to map SOP UIDs to slice indices
            sop_to_index = {}
            
            # Get SOPInstanceUID for each DICOM file
            for i, dcm_path in enumerate(dicom_files):
                try:
                    ds = pydicom.dcmread(dcm_path)
                    sop_to_index[ds.SOPInstanceUID] = i
                except Exception as e:
                    print(f"Warning: Could not read DICOM file {dcm_path}: {str(e)}")
            
            # Process each nodule
            for nodule in nodules:
                # Process each ROI
                rois = nodule.findall('.//ns:roi', ns)
                if not rois:
                    # Try without namespace
                    rois = nodule.findall('.//roi')
                
                for roi in rois:
                    # Get SOP UID to identify the slice
                    sop_uid_elem = roi.find('ns:imageSOP_UID', ns)
                    if sop_uid_elem is None:
                        sop_uid_elem = roi.find('imageSOP_UID')
                    
                    if sop_uid_elem is None:
                        continue
                        
                    sop_uid = sop_uid_elem.text
                    
                    # Skip if we can't find this SOP UID
                    if sop_uid not in sop_to_index:
                        continue
                    
                    # Get slice index
                    slice_idx = sop_to_index[sop_uid]
                    
                    # Extract edge points
                    points = []
                    edge_maps = roi.findall('.//ns:edgeMap', ns)
                    if not edge_maps:
                        edge_maps = roi.findall('.//edgeMap')
                    
                    for edge in edge_maps:
                        x_elem = edge.find('ns:xCoord', ns)
                        y_elem = edge.find('ns:yCoord', ns)
                        
                        if x_elem is None:
                            x_elem = edge.find('xCoord')
                        if y_elem is None:
                            y_elem = edge.find('yCoord')
                        
                        if x_elem is not None and y_elem is not None:
                            x = int(x_elem.text)
                            y = int(y_elem.text)
                            points.append((x, y))
                    
                    # Create binary mask for this slice if we have at least 3 points (for a polygon)
                    if len(points) >= 3:
                        # Create a blank image
                        img = Image.new('L', (original_shape[2], original_shape[1]), 0)
                        draw = ImageDraw.Draw(img)
                        
                        # Draw polygon
                        draw.polygon(points, fill=1)
                        
                        # Convert to numpy array and add to the volume
                        slice_mask = np.array(img)
                        mask_volume[slice_idx] = np.maximum(mask_volume[slice_idx], slice_mask)
                    elif len(points) == 2:
                        # Create a blank image
                        img = Image.new('L', (original_shape[2], original_shape[1]), 0)
                        draw = ImageDraw.Draw(img)
                        
                        # Draw a line (giving it some thickness)
                        draw.line(points, fill=1, width=3)
                        
                        # Convert to numpy array and add to the volume
                        slice_mask = np.array(img)
                        mask_volume[slice_idx] = np.maximum(mask_volume[slice_idx], slice_mask)
                    elif len(points) == 1:
                        # Create a blank image
                        img = Image.new('L', (original_shape[2], original_shape[1]), 0)
                        draw = ImageDraw.Draw(img)
                        
                        # Draw a point (small circle)
                        x, y = points[0]
                        draw.ellipse((x-3, y-3, x+3, y+3), fill=1)
                        
                        # Convert to numpy array and add to the volume
                        slice_mask = np.array(img)
                        mask_volume[slice_idx] = np.maximum(mask_volume[slice_idx], slice_mask)
            
            # Resize the mask after creation
            mask_volume = self._resize_mask(mask_volume, original_shape)
            return mask_volume
                
        except Exception as e:
            print(f"Error parsing XML for mask creation: {str(e)}")
            # Resize the zero mask in case of error
            mask_volume = self._resize_mask(mask_volume, original_shape)
            return mask_volume
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            tuple: (volume, mask, diagnosis)
                - volume (torch.Tensor): 3D volume of shape (slices, height, width)
                - mask (torch.Tensor): Segmentation mask of the same shape
                - diagnosis (int): Diagnosis class/label
        """
        path, diagnosis = self.samples[idx]
        
        # Load the volume from DICOM files
        try:
            volume = self._load_volume(path)
            mask = self._load_mask(path)
            
            # Convert to tensors
            volume_tensor = torch.from_numpy(volume).float()
            mask_tensor = torch.from_numpy(mask).float()
            
            # Apply transformations if any
            if self.transform:
                volume_tensor = self.transform(volume_tensor)
            
            # Print the shape and diagnosis information
            print(f"SHAPE: {volume.shape[0]} X {volume.shape[1]} X {volume.shape[2]}. DIAGNOSIS: {diagnosis}")
            
            return volume_tensor, mask_tensor, diagnosis
            
        except Exception as e:
            print(f"Error loading sample {idx} from {path}: {str(e)}")
            # Return a dummy sample in case of error
            dummy_depth = 10 if not self.resize_slices else 5
            return torch.zeros((dummy_depth, self.target_size[0], self.target_size[1])), torch.zeros((dummy_depth, self.target_size[0], self.target_size[1])), diagnosis


# Example usage
if __name__ == "__main__":
    # Create the dataset
    dataset = LungNoduleDataset(
        paths_file="paths_with_diagnosis.txt",
        target_size=(256, 256),  # Resize to 256×256
        resize_slices=False      # Keep original number of slices
    )
    
    # Get a sample
    volume, mask, diagnosis = dataset[0]
    
    print(f"Volume shape: {volume.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Diagnosis: {diagnosis}")
    
    # Memory usage estimate (assuming float32)
    memory_mb = (volume.numel() * 4) / (1024 * 1024)
    print(f"Approximate memory usage per volume: {memory_mb:.2f} MB")