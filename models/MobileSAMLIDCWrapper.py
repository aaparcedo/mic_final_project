from mobile_sam import sam_model_registry
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SAM_CHECKPOINT_PATH = "/lustre/fs1/home/cap5516.student2/mic_final_project/MobileSAM/weights/mobile_sam.pt"

class MobileSAMLIDCWrapper(nn.Module):
    def __init__(self, sam_checkpoint_path=SAM_CHECKPOINT_PATH, device="cuda", num_classes=4):
        super().__init__()
        self.device = device
        self.mobile_sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint_path).to(device)
        
        # Freeze original SAM
        for p in self.mobile_sam.parameters():
            p.requires_grad = False
            
        # Unfreeze specific blocks in the image encoder (like in the working implementation)
        for name, param in self.mobile_sam.image_encoder.named_parameters():
            if "blocks.2" in name or "blocks.3" in name or "blocks.4" in name or "blocks.5" in name:
                param.requires_grad = True
        
        # Classification head with bottleneck
        self.bottleneck = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)
        
        # Segmentation head similar to the working implementation
        self.head = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        # Save original input size for later resizing
        original_size = (x.shape[2], x.shape[3])
        batch_size = x.shape[0]
        
        # Ensure input is correctly sized for SAM
        if x.shape[-2:] != (1024, 1024):
            x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        # Ensure we have 3 channels for SAM encoder
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Extract features using SAM encoder
        image_embeddings = self.mobile_sam.image_encoder(x)
        
        # Get sparse and dense embeddings without using mask-based centroids
        sparse_embeddings, dense_embeddings = self.mobile_sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None
        )
        
        # Get low-res masks from SAM's mask decoder
        low_res_masks, _ = self.mobile_sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.mobile_sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # Apply our custom segmentation head to refine the masks
        logits = self.head(low_res_masks)
        
        # Resize masks to original input size
        logits = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
        
        masks = torch.sigmoid(logits)
        
        # Classification output
        bottleneck_features = self.bottleneck(image_embeddings)
        bottleneck_flat = bottleneck_features.view(bottleneck_features.size(0), -1)
        cls_output = self.classifier(bottleneck_flat)
        
        return masks, cls_output