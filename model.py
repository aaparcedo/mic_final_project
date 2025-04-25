import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.PMFSNet import PMFSNetWithClassifier
from models.MobileSAMLIDCWrapper import MobileSAMLIDCWrapper
import segmentation_models_pytorch as smp

def create_model(model_type, device):
    if model_type == "UNet":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1, # segmentation classes
            aux_params={'classes': 4, 'dropout': 0.3, 'activation': None} # classification classes
        ).to(device)

    elif model_type == "MobileSAM":
        return MobileSAMLIDCWrapper(
            device=device,
            num_classes=4 # classification classes
        ).to(device)
        

    elif model_type == "PMFSNet":
        return PMFSNetWithClassifier(
            in_channels=1,
            out_channels=1, # segmentation classes
            num_classes=4, # classification classes
            dim="2d",
            scaling_version="TINY"
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


