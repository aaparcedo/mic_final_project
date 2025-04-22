# from dataset import LungNoduleDatasetSliced
# from torchvision.transforms import Resize
# from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# import torch
# import torch.nn as nn
# import torch.F.functional as F

# class MobileSAMLIDCWrapper(nn.Module):
#     def __init__(self, sam_checkpoint_path, device="cuda", mode='2D'):
#         super().__init__()
#         self.device = device
#         self.mode = mode
#         self.sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint_path).to(device)

#         # Freeze original parameters
#         for param in self.sam.parameters():
#             param.requires_grad_(False)

#         # Add trainable components
#         self.adaptor = nn.Sequential(
#             nn.Conv2d(256, 128, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 64, 3, padding=1),
#             nn.Upsample(scale_factor=2, mode='bilinear')
#         )

#         # Configure based on modality
#         if mode == '3D':
#             self.proj_3d = nn.Conv3d(1, 3, kernel_size=(16, 3, 3), padding=(8, 1, 1))

#     def forward(self, x):
#         if self.mode == '3D':
#             # Process 3D volumes slice-wise
#             batch_size = x.shape[0]
#             outputs = []
#             for i in range(x.shape[2]):  # Iterate through slices
#                 slice = x[:, :, :, i].unsqueeze(1)  # [B, 1, H, W]
#                 if slice.shape[-2:] != (1024, 1024):
#                     slice = F.interpolate(slice, size=1024, mode='bilinear')

#                 # SAM processing
#                 with torch.no_grad():
#                     features = self.sam.image_encoder(slice.repeat(1, 3, 1, 1))

#                 # Feature adaptation
#                 out = self.adaptor(features)
#                 outputs.append(out)

#             # Combine slice features
#             return torch.stack(outputs, dim=2)  # [B, C, D, H, W]

#         elif self.mode == '2D':
#             # Direct 2D processing
#             if x.shape[-2:] != (1024, 1024):
#                 x = F.interpolate(x, size=1024, mode='bilinear')

#             with torch.no_grad():
#                 features = self.sam.image_encoder(x.repeat(1, 3, 1, 1))

#             return self.adaptor(features)


# class SAMEnhancedLIDCDataset(LungNoduleDatasetSliced):
#     def __init__(self, *args, sam_wrapper=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.sam_wrapper = sam_wrapper
#         self.resize = Resize((1024, 1024)) if self.mode == '2D' else None

#     def __getitem__(self, idx):
#         volume, mask, diagnosis = super().__getitem__(idx)

#         if self.sam_wrapper:
#             with torch.no_grad():
#                 if self.mode == '2D':
#                     sam_features = self.sam_wrapper(volume.unsqueeze(0))
#                 else:
#                     sam_features = self.sam_wrapper(volume.permute(1, 0, 2, 3).unsqueeze(0))

#             # Fuse SAM features with original data
#             volume = torch.cat([F.interpolate(volume.unsqueeze(0), sam_features, dim=1)]).squeeze(0)

#         return volume, mask, diagnosis