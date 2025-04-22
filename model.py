import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.PMFSNet import PMFSNetWithClassifier
# from models.MobileSAMLIDCWrapper import MobileSAMLIDCWrapper
import segmentation_models_pytorch as smp


# class LungNoduleNetSliced(nn.Module):
#     """
#     Combined model for nodule segmentation and classification
#     """

#     def __init__(self, in_channels=1, out_channels=1, num_classes=6, roi_size=(64, 64, 64), margin=16):
#         super(LungNoduleNetSliced, self).__init__()
#         self.segmentation_net = AttentionUNet3D(in_channels, out_channels)
#         self.classification_head = ClassificationHead(
#             self.segmentation_net.bottleneck_features, num_classes
#         )
#         self.roi_size = roi_size
#         self.margin = margin  # Extra space around nodule

#         self._initialize_weights()

#     def forward(self, x):
#         # Get segmentation and bottleneck features
#         seg_logits, bottleneck = self.segmentation_net(x)

#         # Get classification logits
#         cls_logits = self.classification_head(bottleneck)

#         return seg_logits, cls_logits

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 # Very conservative initialization to prevent exploding gradients
#                 nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 0.1)  # Start with smaller batch norm weights
#                 nn.init.constant_(m.bias, 0)

#     def _find_roi_centroid(self, mask):
#         # Find largest connected component
#         labeled_mask, num_labels = label(mask)
#         if num_labels == 0:
#             return None  # No nodule

#         largest_cc = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
#         z, y, x = np.where(labeled_mask == largest_cc)
#         centroid = (
#             (z.max() + z.min()) // 2,
#             (y.max() + y.min()) // 2,
#             (x.max() + x.min()) // 2
#         )
#         return centroid

#     def __getitem__(self, idx):
#         volume, mask, diagnosis = super().__getitem__(idx)

#         # Extract ROI around nodule
#         centroid = self._find_roi_centroid(mask.numpy())
#         if centroid:
#             z_start = max(0, centroid[0] - self.roi_size[0] // 2 - self.margin)
#             y_start = max(0, centroid[1] - self.roi_size[1] // 2 - self.margin)
#             x_start = max(0, centroid[2] - self.roi_size[2] // 2 - self.margin)

#             # Apply cropping
#             volume = volume[z_start:z_start + self.roi_size[0],
#                      y_start:y_start + self.roi_size[1],
#                      x_start:x_start + self.roi_size[2]]
#             mask = mask[z_start:z_start + self.roi_size[0],
#                    y_start:y_start + self.roi_size[1],
#                    x_start:x_start + self.roi_size[2]]
#         return volume, mask, diagnosis


def create_model(model_type, device):
    if model_type == "UNet":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1, # segmentation classes
            aux_params={'classes': 4, 'dropout': 0.3, 'activation': None} # classification classes
        ).to(device)

    # elif model_type == "MobileSAM":
    #     model = MobileSAMLIDCWrapper(
    #         sam_checkpoint_path="mobile_sam.pth",
    #         device=device,
    #         mode='2D'
    #     )
    #     # Add final conv layer for segmentation output
    #     model.final_conv = nn.Conv2d(64, 1, kernel_size=1).to(device)
    #     return model

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


