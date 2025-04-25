import segmentation_models_pytorch as smp
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset2d import LIDCIDRIAugmentedDataset
import torch.optim as optim
import time
import os
from model import create_model
import wandb
from utils import measure_inference_time, print_trainable_parameters

run_name = "unet"

wandb.init(project="mic_final_project", name=run_name)

wandb.config.model_type = "UNet"  # Can be UNet/MobileSAM/PMFSNet
wandb.config.epochs = 20
wandb.config.batch_size = 16
wandb.config.seed = 42
wandb.config.learning_rate = 1e-3
wandb.config.use_augmentation = True

if wandb.config.model_type == "UNet":
    wandb.config.encoder_name = "resnet34"
    wandb.config.encoder_weights = "imagenet"

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

train_dataset = LIDCIDRIAugmentedDataset(
    split='train',
    use_augmented=wandb.config.use_augmentation
)

val_dataset = LIDCIDRIAugmentedDataset(
    split='val',
    use_augmented=wandb.config.use_augmentation
)

test_dataset = LIDCIDRIAugmentedDataset(
    split='test',
    use_augmented=wandb.config.use_augmentation
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

model = create_model(wandb.config.model_type, device)

print_trainable_parameters(model)

total_samples = 101 + 224 + 658 + 573  # 1556
num_classes = 4
class_samples = [101, 224, 658, 573]
class_weights = torch.FloatTensor([total_samples / (num_classes * count) for count in class_samples]).to(device)

# loss, optimizer, scheduler
if wandb.config.model_type == 'MobileSAM':
    seg_criterion = smp.losses.DiceLoss(mode='binary', from_logits=False)
    optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
else:
    seg_criterion = smp.losses.DiceLoss(mode='binary')    
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
cls_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

wandb.config.update({
    "seg_loss": seg_criterion.__class__.__name__,
    "cls_loss": cls_criterion.__class__.__name__,
    "optimizer": optimizer.__class__.__name__,
    "weight_decay": optimizer.param_groups[0].get('weight_decay', 0),
    "input_size": {
        "UNet": (512, 512),
        "MobileSAM": (1024, 1024),
        "PMFSNet": (256, 256)},
    "normalization": "model-specific",
    "architecture": wandb.config.model_type
})

best_val_loss = float('inf')

early_stopper = 0
early_stopping_limit = 10

# print("Measuring inference time...")
# inference_time = measure_inference_time(model, test_loader, device)

for epoch in range(NUM_EPOCHS):
    
    # Training phase
    model.train()
    train_loss = 0.0
    
    train_progress = tqdm(train_loader, desc="Training")
    
    # training loop
    for batch_idx, batch in enumerate(train_progress):

        image = batch["image"].to(device) # add channel dimension, image.shape = [B, 1, 512, 512]
        mask = batch["mask"].to(device) # add channel dimension, mask.shape = [B, 1, 512, 512]
        diagnosis = batch["diagnosis"].to(device) # diagnosis.shape = [B, 1]
                
        optimizer.zero_grad()

        if wandb.config.model_type == 'MobileSAM':
            seg_preds, cls_preds = model(image, mask=mask) 
        else:
            seg_preds, cls_preds = model(image) # seg_preds.shape (B, C, H, W), cls_preds.shape [B]
            
        seg_loss = seg_criterion(seg_preds, mask)
        cls_loss = cls_criterion(cls_preds, diagnosis)
        
        combined_loss = seg_loss + cls_loss
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
            image = batch["image"].to(device) # add channel dimension, image.shape = [B, 1, 512, 512]
            mask = batch["mask"].to(device) # add channel dimension, mask.shape = [B, 1, 512, 512]
            diagnosis = batch["diagnosis"].to(device) # diagnosis.shape = [B, 1]
                        
            seg_preds, cls_preds = model(image) # seg_preds.shape (B, C, H, W), cls_preds.shape [B, NUM_CLASSES]
            
            seg_loss = seg_criterion(seg_preds, mask)
            cls_loss = cls_criterion(cls_preds, diagnosis)
            
            combined_loss = (seg_loss + cls_loss)
            
            val_loss += combined_loss.item()            
            val_progress.set_postfix({'val loss': (val_loss / (batch_idx + 1))})
            
            tp, fp, fn, tn = smp.metrics.get_stats(seg_preds, mask.round().long(), mode='binary', threshold=0.5)
            
            val_total_dice_score += smp.metrics.f1_score(tp, fp, fn, tn).mean()
            val_total_iou_score += smp.metrics.iou_score(tp, fp, fn, tn).mean()
            val_total_recall_score += smp.metrics.recall(tp, fp, fn, tn).mean()
            val_total_precision_score += smp.metrics.precision(tp, fp, fn, tn).mean()
            
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
            output_dir = "/home/cap5516.student2/mic_final_project/output"
            ckpt_name = f'{run_name}_epoch{epoch}_{time.strftime("%Y%m%d-%H%M%S")}.pth'
            save_path = os.path.join(output_dir, ckpt_name)
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
    

# LOAD BEST MODEL FOR TESTING
model = create_model(wandb.config.model_type, device)
checkpoint = torch.load(save_path, map_location=torch.device('cuda'))
print(f'Loading best model from run: {save_path}')
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
        image = batch["image"].to(device) # add channel dimension, image.shape = [B, 1, 512, 512]
        mask = batch["mask"].to(device) # add channel dimension, mask.shape = [B, 1, 512, 512]
        diagnosis = batch["diagnosis"].to(device) # diagnosis.shape = [B, 1]
        
        seg_preds, cls_preds = model(image) # seg_preds.shape (B, C, H, W), cls_preds.shape [B, NUM_CLASSES]
        seg_loss = seg_criterion(seg_preds, mask)
        cls_loss = cls_criterion(cls_preds, diagnosis)
        
        test_loss += (seg_loss + cls_loss).item()
        test_progress.set_postfix({'test loss': (test_loss / (batch_idx + 1))})
        
        tp, fp, fn, tn = smp.metrics.get_stats(seg_preds, mask.round().long(), mode='binary', threshold=0.5)
        
        test_total_dice_score += smp.metrics.f1_score(tp, fp, fn, tn).mean()
        test_total_iou_score += smp.metrics.iou_score(tp, fp, fn, tn).mean()
        test_total_recall_score += smp.metrics.recall(tp, fp, fn, tn).mean()
        test_total_precision_score += smp.metrics.precision(tp, fp, fn, tn).mean()
        
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
