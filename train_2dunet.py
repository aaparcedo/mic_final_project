import segmentation_models_pytorch as smp
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset2d import LIDCIDRI2DDataset
import torch.optim as optim
import time
import os

import wandb
wandb.init(project="mic_final_project")

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


model = smp.Unet(
    encoder_name=wandb.config.encoder_name, 
    encoder_weights=wandb.config.encoder_weights,
    in_channels=1, 
    classes=1,
    aux_params={'classes': 4, 'dropout': 0.3, 'activation': None}
).to(device)

total_samples = 101 + 224 + 658 + 573  # 1556
num_classes = 4
class_samples = [101, 224, 658, 573]
class_weights = torch.FloatTensor([total_samples / (num_classes * count) for count in class_samples]).to(device)

# loss, optimizer, schedular
seg_criterion = smp.losses.DiceLoss(mode='binary')
cls_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

wandb.config.update({
    "seg_loss": seg_criterion.__class__.__name__,
    "cls_loss": cls_criterion.__class__.__name__,
    "optimizer": optimizer.__class__.__name__,
    "weight_decay": optimizer.param_groups[0].get('weight_decay', 0),
})

best_val_loss = float('inf')

early_stopper = 0
early_stopping_limit = 10

for epoch in range(NUM_EPOCHS):
    
    # Training phase
    model.train()
    train_loss = 0.0
    
    train_progress = tqdm(train_loader, desc="Training")
    
    # training loop
    for batch_idx, batch in enumerate(train_progress):
        image = batch["image"].unsqueeze(dim=1).to(device) # add channel dimension, image.shape = [B, 1, 512, 512]
        mask = batch["mask"].unsqueeze(dim=1).to(device) # add channel dimension, mask.shape = [B, 1, 512, 512]
        diagnosis = batch["diagnosis"].to(device) # diagnosis.shape = [B, 1]
        
        assert diagnosis.shape == torch.Size([image.shape[0]]), f'diagnosis shape: {diagnosis.shape}, expected: {torch.Size([BATCH_SIZE])}'
        
        optimizer.zero_grad()
        
        seg_preds, cls_preds = model(image) # seg_preds.shape (B, C, H, W), cls_preds.shape [B, NUM_CLASSES]
        seg_loss = seg_criterion(seg_preds, mask)
        cls_loss = cls_criterion(cls_preds, diagnosis)
        
        combined_loss = seg_loss + cls_loss
        combined_loss.backward()
        
        train_loss += combined_loss.item()

        optimizer.step()
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
    val_total_cls_accuracy = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_progress):
            image = batch["image"].unsqueeze(dim=1).to(device) # add channel dimension, image.shape = [B, 1, 512, 512]
            mask = batch["mask"].unsqueeze(dim=1).to(device) # add channel dimension, mask.shape = [B, 1, 512, 512]
            diagnosis = batch["diagnosis"].to(device) # diagnosis.shape = [B, 1]
            
            assert diagnosis.shape == torch.Size([image.shape[0]]), f'diagnosis shape: {diagnosis.shape}, expected: {torch.Size([BATCH_SIZE])}'
            
            seg_preds, cls_preds = model(image) # seg_preds.shape (B, C, H, W), cls_preds.shape [B, NUM_CLASSES]
            seg_loss = seg_criterion(seg_preds, mask)
            cls_loss = cls_criterion(cls_preds, diagnosis)
            
            combined_loss = (seg_loss + cls_loss)
            
            val_loss += combined_loss.item()
            
            val_progress.set_postfix({'val loss': (val_loss / (batch_idx + 1))})
            
            tp, fp, fn, tn = smp.metrics.get_stats(seg_preds, mask.round().long(), mode='binary', threshold=0.5)
            
            val_total_dice_score += smp.metrics.f1_score(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
            val_total_iou_score += smp.metrics.iou_score(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
            val_total_recall_score += smp.metrics.recall(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
            val_total_precision_score += smp.metrics.precision(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
            
            _, predicted_classes = torch.max(cls_preds, dim=1)
            val_total_cls_accuracy += (predicted_classes == diagnosis).float().mean()
            
        val_loss /= len(val_loader)
        val_total_dice_score /= len(val_loader)
        val_total_iou_score /= len(val_loader)
        val_total_recall_score /= len(val_loader)
        val_total_precision_score /= len(val_loader)
        val_total_cls_accuracy /= len(val_loader)
        
        if val_loss < best_val_loss:
            early_stopper = 0
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice_score': val_total_dice_score,
                'val_cls_accuracy': val_total_cls_accuracy,
            }
            output_dir = "/home/cap5516.student2/mic_final_project/output/2dunet"
            run_name = f'2DUnetBaseline_epoch{epoch}_{time.strftime("%Y%m%d-%H%M%S")}.pth'
            save_path = os.path.join(output_dir, run_name)
        else:
            early_stopper += 1
            
        if early_stopper > early_stopping_limit:
            break
    
    print(f'epoch {epoch} train loss: {train_loss:.2f}, val loss: {val_loss:.2f}')
    print(f'cls accuracy: {val_total_cls_accuracy:.2f}')
    print(f'dice: {val_total_dice_score:.2f}, iou: {val_total_iou_score:.2f}, recall: {val_total_recall_score:.2f}, precision: {val_total_precision_score:.2f}')
    wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dice_score": val_total_dice_score,
            "iou_score": val_total_iou_score,
            "recall_score": val_total_recall_score,
            "precision_score": val_total_precision_score,
            "cls_accuracy": val_total_cls_accuracy
            })
    
torch.save(checkpoint, save_path)
    

# LOAD BEST MODEL AND TEST IT
model = smp.Unet(
    encoder_name=wandb.config.encoder_name, 
    encoder_weights=wandb.config.encoder_weights,
    in_channels=1, 
    classes=1,
    aux_params={'classes': 4, 'dropout': 0.3, 'activation': None}
).to(device)
    
checkpoint = torch.load(save_path, map_location=torch.device('cuda'))
model.load_state_dict(checkpoint['model_state_dict'])

test_loss = 0.0
test_total_dice_score = 0.0
test_total_iou_score = 0.0
test_total_recall_score = 0.0
test_total_precision_score = 0.0
test_total_cls_accuracy = 0.0

# test loop
model.eval()
test_progress = tqdm(test_loader, desc="Testing")
with torch.no_grad():
    for batch_idx, batch in enumerate(test_progress):
        image = batch["image"].unsqueeze(dim=1).to(device) # add channel dimension, image.shape = [B, 1, 512, 512]
        mask = batch["mask"].unsqueeze(dim=1).to(device) # add channel dimension, mask.shape = [B, 1, 512, 512]
        diagnosis = batch["diagnosis"].to(device) # diagnosis.shape = [B, 1]
        
        seg_preds, cls_preds = model(image) # seg_preds.shape (B, C, H, W), cls_preds.shape [B, NUM_CLASSES]
        seg_loss = seg_criterion(seg_preds, mask)
        cls_loss = cls_criterion(cls_preds, diagnosis)
        
        test_loss += (seg_loss + cls_loss).item()
        test_progress.set_postfix({'test loss': (test_loss / (batch_idx + 1))})
        
        tp, fp, fn, tn = smp.metrics.get_stats(seg_preds, mask.round().long(), mode='binary', threshold=0.5)
        
        test_total_dice_score += smp.metrics.f1_score(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
        test_total_iou_score += smp.metrics.iou_score(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
        test_total_recall_score += smp.metrics.recall(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
        test_total_precision_score += smp.metrics.precision(tp, fp, fn, tn, reduction=REDUCTION, class_weights=CLASS_WEIGHTS)
        
        _, predicted_classes = torch.max(cls_preds, dim=1)

        test_total_cls_accuracy += (predicted_classes == diagnosis).float().mean()
        
        
    test_loss /= len(test_loader)
    test_total_dice_score /= len(test_loader)
    test_total_iou_score /= len(test_loader)
    test_total_recall_score /= len(test_loader)
    test_total_precision_score /= len(test_loader)
    test_total_cls_accuracy /= len(test_loader)
print(f'test loss: {test_loss}')
print(f'cls accuracy: {test_total_cls_accuracy}')
print(f'dice: {test_total_dice_score:.2f}, iou: {test_total_iou_score:.2f}, recall: {test_total_recall_score:.2f}, precision: {test_total_precision_score:.2f}')
wandb.log({
        "test_loss": test_loss,
        "test_total_dice_score": test_total_dice_score,
        "test_total_iou_score": test_total_iou_score,
        "test_total_recall_score": test_total_recall_score,
        "test_total_precision_score": test_total_precision_score,
        "test_total_cls_accuracy": test_total_cls_accuracy
        })

    
wandb.finish()
