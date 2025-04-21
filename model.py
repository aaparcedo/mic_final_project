import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.PMFSNet import PMFSNet
from models.MobileSAMLIDCWrapper import MobileSAMLIDCWrapper
import segmentation_models_pytorch as smp

class ConvBlock(nn.Module):
    """
    3D Convolutional block with batch normalization and ReLU activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    """
    Downsampling block with max pooling followed by convolution
    """
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class AttentionGate(nn.Module):
    """
    Attention Gate module to focus on relevant regions of feature maps
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class UpSample(nn.Module):
    """
    Upsampling block with attention gate
    """
    def __init__(self, in_channels, out_channels, trilinear=True):
        super(UpSample, self).__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            # After upsampling, the channel count remains the same
            self.conv1x1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
            self.conv = ConvBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)
            
        # FIX: Use proper channel counts for attention gate
        # g (gating signal) has in_channels // 2 channels after 1x1 conv
        # x (skip connection) has out_channels channels
        self.attention = AttentionGate(
            F_g=in_channels // 2,  # Channels of gating signal after 1x1 conv
            F_l=out_channels,      # Channels of skip connection (corresponds to earlier layer output)
            F_int=out_channels // 2 # Intermediate channel dimension for attention
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Reduce channels in x1 after upsampling
        if hasattr(self, 'conv1x1'):
            x1 = self.conv1x1(x1)
        
        # Adjust sizes if needed (for odd dimensions)
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2,
                        diff_z // 2, diff_z - diff_z // 2])
        
        # Apply attention mechanism
        x2 = self.attention(x1, x2)
        
        # Concatenate along the channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution layer
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)


class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net for nodule segmentation
    """
    def __init__(self, in_channels, out_channels, n_channels=(16, 32, 64, 128, 256)):
        super(AttentionUNet3D, self).__init__()
        self.n_channels = n_channels
        
        # Encoder
        self.inc = ConvBlock(in_channels, n_channels[0])
        self.down1 = DownSample(n_channels[0], n_channels[1])
        self.down2 = DownSample(n_channels[1], n_channels[2])
        self.down3 = DownSample(n_channels[2], n_channels[3])
        self.down4 = DownSample(n_channels[3], n_channels[4])
        
        # Decoder
        self.up1 = UpSample(n_channels[4], n_channels[3])
        self.up2 = UpSample(n_channels[3], n_channels[2])
        self.up3 = UpSample(n_channels[2], n_channels[1])
        self.up4 = UpSample(n_channels[1], n_channels[0])
        
        # Output
        self.outc = OutConv(n_channels[0], out_channels)
        
        # Feature extraction for classification
        self.bottleneck_features = n_channels[4]
        
        
        
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Debug print for troubleshooting
        # print(f"Encoder feature shapes: x1={x1.shape}, x2={x2.shape}, x3={x3.shape}, x4={x4.shape}, x5={x5.shape}")
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output segmentation
        logits = self.outc(x)
        
        # Return both segmentation logits and bottleneck features for classification
        return logits, x5


class ClassificationHead(nn.Module):
    """
    Classification head that takes bottleneck features from U-Net
    """
    def __init__(self, in_features, num_classes):
        super(ClassificationHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LungNoduleNet(nn.Module):
    """
    Combined model for nodule segmentation and classification
    """
    def __init__(self, in_channels=1, out_channels=1, num_classes=6):
        super(LungNoduleNet, self).__init__()
        self.segmentation_net = AttentionUNet3D(in_channels, out_channels)
        self.classification_head = ClassificationHead(
            self.segmentation_net.bottleneck_features, num_classes
        )
        
        self._initialize_weights()
        
    def forward(self, x):
        # Get segmentation and bottleneck features
        seg_logits, bottleneck = self.segmentation_net(x)
        
        # Get classification logits
        cls_logits = self.classification_head(bottleneck)
        
        return seg_logits, cls_logits
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Very conservative initialization to prevent exploding gradients
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 0.1)  # Start with smaller batch norm weights
                nn.init.constant_(m.bias, 0)


# Define loss functions
class CombinedLoss(nn.Module):
    """
    Combined loss function for joint segmentation and classification
    """
    def __init__(self, lambda_dice=1.0, lambda_focal=1.0, lambda_cls=1.0):
        super(CombinedLoss, self).__init__()
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.lambda_cls = lambda_cls
        self.cls_loss = nn.CrossEntropyLoss()
        
    def dice_loss(self, pred, target):
        """Numerically stable dice loss"""
        smooth = 1e-6  # Increase smoothing factor
        
        # Use stable sigmoid
        pred = torch.clamp(torch.sigmoid(pred), min=smooth, max=1.0 - smooth)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum() + smooth
        denominator = pred_flat.sum() + target_flat.sum() + smooth
        
        # Print values for debugging if needed
        # print(f"Intersection: {intersection.item()}, Denominator: {denominator.item()}")
        
        # Avoid division by very small numbers
        # if denominator.item() < smooth * 2:
        #     return torch.tensor(0.0, device=pred.device)
        
        dice_score = (2. * intersection) / denominator
        return 1.0 - dice_score
    
    def focal_loss(self, pred, target, alpha=0.8, gamma=2.0):
        """Focal loss to focus on harder examples"""
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Apply focal weighting
        pt = torch.exp(-bce)
        focal_weight = alpha * (1-pt)**gamma
        
        return (focal_weight * bce).mean()
        
    # In your CombinedLoss class forward method
    def forward(self, seg_logits, cls_logits, seg_targets, cls_targets):
        # For BCE loss, both inputs should be half but targets might need to be float32
        # (depending on the implementation)
        focal = self.focal_loss(seg_logits, seg_targets.float() if seg_targets.dtype != seg_logits.dtype else seg_targets)
        
        # For Dice loss
        dice = self.dice_loss(seg_logits, seg_targets)
        
        # classify as 0: unknown, 1: benign or non-malignant disease, 2: malignant, primary lung cancer, 3: malignant metastatic
        cls = self.cls_loss(cls_logits, cls_targets) 
        
        # Combined loss
        total_loss = (self.lambda_dice * dice + 
                    self.lambda_focal * focal + 
                    self.lambda_cls * cls)
        
        return total_loss, {
            'total': total_loss.item(),
            'dice': dice.item(),
            'focal': focal.item(),
            'cls': cls.item()
        }


class LungNoduleNetSliced(nn.Module):
    """
    Combined model for nodule segmentation and classification
    """

    def __init__(self, in_channels=1, out_channels=1, num_classes=6, roi_size=(64, 64, 64), margin=16):
        super(LungNoduleNetSliced, self).__init__()
        self.segmentation_net = AttentionUNet3D(in_channels, out_channels)
        self.classification_head = ClassificationHead(
            self.segmentation_net.bottleneck_features, num_classes
        )
        self.roi_size = roi_size
        self.margin = margin  # Extra space around nodule

        self._initialize_weights()

    def forward(self, x):
        # Get segmentation and bottleneck features
        seg_logits, bottleneck = self.segmentation_net(x)

        # Get classification logits
        cls_logits = self.classification_head(bottleneck)

        return seg_logits, cls_logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Very conservative initialization to prevent exploding gradients
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 0.1)  # Start with smaller batch norm weights
                nn.init.constant_(m.bias, 0)

    def _find_roi_centroid(self, mask):
        # Find largest connected component
        labeled_mask, num_labels = label(mask)
        if num_labels == 0:
            return None  # No nodule

        largest_cc = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
        z, y, x = np.where(labeled_mask == largest_cc)
        centroid = (
            (z.max() + z.min()) // 2,
            (y.max() + y.min()) // 2,
            (x.max() + x.min()) // 2
        )
        return centroid

    def __getitem__(self, idx):
        volume, mask, diagnosis = super().__getitem__(idx)

        # Extract ROI around nodule
        centroid = self._find_roi_centroid(mask.numpy())
        if centroid:
            z_start = max(0, centroid[0] - self.roi_size[0] // 2 - self.margin)
            y_start = max(0, centroid[1] - self.roi_size[1] // 2 - self.margin)
            x_start = max(0, centroid[2] - self.roi_size[2] // 2 - self.margin)

            # Apply cropping
            volume = volume[z_start:z_start + self.roi_size[0],
                     y_start:y_start + self.roi_size[1],
                     x_start:x_start + self.roi_size[2]]
            mask = mask[z_start:z_start + self.roi_size[0],
                   y_start:y_start + self.roi_size[1],
                   x_start:x_start + self.roi_size[2]]
        return volume, mask, diagnosis


def create_model(model_type, device):
    if model_type == "UNet":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        ).to(device)

    elif model_type == "MobileSAM":
        model = MobileSAMLIDCWrapper(
            sam_checkpoint_path="mobile_sam.pth",
            device=device,
            mode='2D'
        )
        # Add final conv layer for segmentation output
        model.final_conv = nn.Conv2d(64, 1, kernel_size=1).to(device)
        return model

    elif model_type == "PMFSNet":
        return PMFSNet(
            in_channels=1,
            out_channels=1,
            dim="2d",
            scaling_version="TINY"
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


