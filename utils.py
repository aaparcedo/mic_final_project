import torchio as tio

aug_transform = tio.Compose([
    tio.RandomFlip(axes=(0, 1, 2),  # 3D flipping
    tio.RandomAffine(scales=(0.8, 1.2),  # Elastic deformation
    tio.RandomNoise(std=0.1),  # Gaussian noise
    tio.RandomBiasField(coefficients=0.3),  # Intensity variation
    tio.RandomBlur(std=(0, 2)),  # Smoothing
    tio.ZNormalization(),  # Intensity normalization
])