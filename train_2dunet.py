import segmentation_models_pytorch as smp
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset2d import LIDCIDRI2DDataset
import torch.optim as optim
import time
import os
from model import create_model

import wandb

from models.MobileSAMLIDCWrapper import MobileSAMLIDCWrapper

wandb.init(project="mic_final_project")
wandb.config.model_type = "PMFSNet"  # Can be UNet/MobileSAM/PMFSNet
model = create_model(wandb.config.model_type, device)

wandb.config.epochs = 100
wandb.config.batch_size = 32
wandb.config.seed = 42
wandb.config.learning_rate = 1e-3
wandb.config.encoder_name = "resnet34"
wandb.config.encoder_weights = "imagenet"
wandb.config.reduction = 'micro'
wandb.config.class_weights = None # [0.0005, 0.9995]

CLASS_WEIGHTS = wandb.config.class_weights
REDUCTION = wandb.config.reduction
NUM_EPOCHS = wandb.config.epochs
BATCH_SIZE = wandb.config.batch_size


# Modified forward pass handling
def get_segmentation_output(model, images):
    if isinstance(model, MobileSAMLIDCWrapper):
        features = model(images)
        return model.final_conv(features)
    return model(images)

# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(wandb.config.seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = wandb.config.learning_rate
DATASET_DIR = '/home/cap5516.student2/LIDC-IDRI-2D'


train_dataset = LIDCIDRI2DDataset(
    root_dir=DATASET_DIR,
    split='train'
)

val_dataset = LIDCIDRI2DDataset(
    root_dir=DATASET_DIR,
    split='val'
)

test_dataset = LIDCIDRI2DDataset(
    root_dir=DATASET_DIR,
    split='test'
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)


# pretrained or train from scratch?
model = smp.Unet(
    encoder_name=wandb.config.encoder_name, 
    encoder_weights=wandb.config.encoder_weights,
    in_channels=1, 
    classes=1,
).to(device)

# loss, optimizer, schedular
criterion = smp.losses.DiceLoss(mode='binary')
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

wandb.config.update({
    "loss": criterion.__class__.__name__,
    "optimizer": optimizer.__class__.__name__,
    "weight_decay": optimizer.param_groups[0].get('weight_decay', 0),
    "input_size": {
        "UNet": (512, 512),
        "MobileSAM": (1024, 1024),
        "PMFSNet": (256, 256)
    }[wandb.config.model_type],
    "normalization": "model-specific",
    "architecture": wandb.config.model_type
})

best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    
    # Training phase
    model.train()
    train_loss = 0.0
    
    train_progress = tqdm(train_loader, desc="Training")
    
    # training loop
    for batch_idx, batch in enumerate(train_progress):
        images = batch["image"].unsqueeze(dim=1).to(device) # add channel dimension, image.shape = [B, 1, 512, 512]
        masks = batch["mask"].unsqueeze(dim=1).to(device) # add channel dimension, mask.shape = [B, 1, 512, 512]
        # Model-specific forward
        if wandb.config.model_type == "PMFSNet":
            logits = model(images)
        else:
            logits = get_segmentation_output(model, images)
        # logits = model(images) # logits.shape (B, C, H, W)
        loss = criterion(logits, masks)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_progress.set_postfix({'train loss': (train_loss / (batch_idx + 1))})
    train_loss /= len(train_loader)
    
    # validation loop
    model.eval()
    val_progress = tqdm(val_loader, desc="Validation")
    val_loss = 0.0
    val_total_dice_score = 0.0
    val_total_iou_score = 0.0
    val_total_recall_score = 0.0
    val_total_precision_score = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_progress):
            images = batch["image"].unsqueeze(dim=1).to(device) # add channel dimension, image.shape = [B, 1, 512, 512]
            masks = batch["mask"].unsqueeze(dim=1).to(device) # add channel dimension, mask.shape = [B, 1, 512, 512]
            
            logits = model(images) # logits.shape (B, C, H, W)
            loss = criterion(logits, masks)
            
            val_loss += loss.item()
            val_progress.set_postfix({'val loss': (val_loss / (batch_idx + 1))})
            
            tp, fp, fn, tn = smp.metrics.get_stats(logits, masks.round().long(), mode='binary', threshold=0.5)
            
            val_total_dice_score += smp.metrics.f1_score(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
            val_total_iou_score += smp.metrics.iou_score(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
            val_total_recall_score += smp.metrics.recall(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
            val_total_precision_score += smp.metrics.precision(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
            
        val_loss /= len(val_loader)
        val_total_dice_score /= len(val_loader)
        val_total_iou_score /= len(val_loader)
        val_total_recall_score /= len(val_loader)
        val_total_precision_score /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            output_dir = "/home/cap5516.student2/mic_final_project/output/2dunet"
            run_name = f'2DUnetBaseline_epoch{epoch}_{time.strftime("%Y%m%d-%H%M%S")}.pth'
            save_path = os.path.join(output_dir, run_name) 
    
    print(f'epoch {epoch} train loss: {train_loss:.2f}, val loss: {val_loss:.2f}')
    print(f'dice: {val_total_dice_score:.2f}, iou: {val_total_iou_score:.2f}, recall: {val_total_recall_score:.2f}, precision: {val_total_precision_score:.2f}')
    wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dice_score": val_total_dice_score,
            "iou_score": val_total_iou_score,
            "recall_score": val_total_recall_score,
            "precision_score": val_total_precision_score
            })
    
torch.save(checkpoint, save_path)
    

# LOAD BEST MODEL AND TEST IT


model = smp.Unet(
    encoder_name=wandb.config.encoder_name, 
    encoder_weights=wandb.config.encoder_weights,
    in_channels=1, 
    classes=1,
).to(device)
    
checkpoint = torch.load(save_path, map_location=torch.device('cuda'))
model.load_state_dict(checkpoint['model_state_dict'])

test_loss = 0.0
test_total_dice_score = 0.0
test_total_iou_score = 0.0
test_total_recall_score = 0.0
test_total_precision_score = 0.0

# test loop
model.eval()
test_progress = tqdm(test_loader, desc="Testing")
with torch.no_grad():
    for batch_idx, batch in enumerate(test_progress):
        images = batch["image"].unsqueeze(dim=1).to(device) # add channel dimension, image.shape = [B, 1, 512, 512]
        masks = batch["mask"].unsqueeze(dim=1).to(device) # add channel dimension, mask.shape = [B, 1, 512, 512]
        
        logits = model(images) # logits.shape (B, C, H, W)
        loss = criterion(logits, masks)
        
        test_loss += loss.item()
        test_progress.set_postfix({'test loss': (test_loss / (batch_idx + 1))})
        
        tp, fp, fn, tn = smp.metrics.get_stats(logits, masks.round().long(), mode='binary', threshold=0.5)
        
        test_total_dice_score += smp.metrics.f1_score(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
        test_total_iou_score += smp.metrics.iou_score(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
        test_total_recall_score += smp.metrics.recall(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
        test_total_precision_score += smp.metrics.precision(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
        
        
    test_loss /= len(test_loader)
    test_total_dice_score /= len(test_loader)
    test_total_iou_score /= len(test_loader)
    test_total_recall_score /= len(test_loader)
    test_total_precision_score /= len(test_loader)
    
print(f'test loss: {test_loss}')
print(f'dice: {test_total_dice_score:.2f}, iou: {test_total_iou_score:.2f}, recall: {test_total_recall_score:.2f}, precision: {test_total_precision_score:.2f}')
wandb.log({
        "test_loss": test_loss,
        "test_total_dice_score": test_total_dice_score,
        "test_total_iou_score": test_total_iou_score,
        "test_total_recall_score": test_total_recall_score,
        "test_total_precision_score": test_total_precision_score
        })

    
wandb.finish()
