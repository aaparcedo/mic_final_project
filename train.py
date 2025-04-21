import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
from tqdm import tqdm
from dataset import LungNoduleDataset
from model import LungNoduleNet, CombinedLoss


# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Load dataset
    dataset = LungNoduleDataset(
        paths_file=args.paths_file,
        transform=None  # Add transforms if needed
    )
    
    # Split dataset into train and validation sets
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Total samples: {dataset_size}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # Drop last batch if incomplete (helpful for BatchNorm)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = LungNoduleNet(
        in_channels=1,  # Assuming single-channel input (CT scans)
        out_channels=1,  # Binary segmentation (nodule/background)
        num_classes=args.num_classes
    ).to(device)
    
    print_model_parameters(model)
    
    # Enable synchronized BatchNorm for multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Loss function and optimizer
    criterion = CombinedLoss(
        lambda_dice=args.lambda_dice,
        lambda_focal=args.lambda_focal,
        lambda_cls=args.lambda_cls
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    scaler = GradScaler(enabled=args.use_amp)
    
    best_val_loss = float('inf')
    
    accum_iter = args.gradient_accumulation_steps
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice_loss = 0.0
        train_focal_loss = 0.0
        train_cls_loss = 0.0
        
        train_progress = tqdm(train_loader, desc="Training")
        
        for batch_idx, (volumes, masks, diagnoses) in enumerate(train_progress):
            # Move data to device
            volumes = volumes.unsqueeze(1).to(device, non_blocking=True)
            masks = masks.unsqueeze(1).float().to(device, non_blocking=True)
            diagnoses = diagnoses.long().to(device, non_blocking=True)
            
            # Zero the parameter gradients if it's the start of a new accumulation cycle
            if (batch_idx % accum_iter == 0) or accum_iter == 1:
                optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast(enabled=args.use_amp, device_type='cuda'):
                seg_logits, cls_logits = model(volumes)
                loss, loss_components = criterion(seg_logits, cls_logits, masks, diagnoses)
                # Normalize loss for gradient accumulation (if used)
                if accum_iter > 1:
                    loss = loss / accum_iter
                    
            scaler.scale(loss).backward()
            
            # Update weights if it's the end of an accumulation cycle or the last batch
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                # Gradient clipping (optional but helpful for stability)
                if args.clip_gradients:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
                
                scaler.step(optimizer)
                scaler.update()
            
            train_loss += loss_components['total']
            train_dice_loss += loss_components['dice']
            train_focal_loss += loss_components['focal']
            train_cls_loss += loss_components['cls']
            
            train_progress.set_postfix({
                'loss': loss_components['total'],
                'dice': loss_components['dice'],
                'focal': loss_components['focal'],
                'cls': loss_components['cls']
            })
            
            torch.cuda.empty_cache()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        train_dice_loss /= len(train_loader)
        train_focal_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice_loss = 0.0
        val_focal_loss = 0.0
        val_cls_loss = 0.0
        
        val_progress = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch_idx, (volumes, masks, diagnoses) in enumerate(val_progress):
                volumes = volumes.unsqueeze(1).to(device, non_blocking=True)  
                masks = masks.unsqueeze(1).float().to(device, non_blocking=True)
                diagnoses = diagnoses.long().to(device, non_blocking=True)
                
                with autocast(enabled=args.use_amp, device_type='cuda'):
                    seg_logits, cls_logits = model(volumes)
                    loss, loss_components = criterion(seg_logits, cls_logits, masks, diagnoses)
                    
                val_loss += loss_components['total']
                val_dice_loss += loss_components['dice']
                val_focal_loss += loss_components['focal']
                val_cls_loss += loss_components['cls']
                
                
                val_progress.set_postfix({
                    'loss': loss_components['total'],
                    'dice': loss_components['dice'],
                    'focal': loss_components['focal'],
                    'cls': loss_components['cls']
                })
                
                torch.cuda.empty_cache()
        
        # Calculate average losses
        val_loss /= len(val_loader)
        val_dice_loss /= len(val_loader)
        val_focal_loss /= len(val_loader)
        val_cls_loss /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Dice: {train_dice_loss:.4f} | Val Dice: {val_dice_loss:.4f}")
        print(f"Train focal: {train_focal_loss:.4f} | Val focal: {val_focal_loss:.4f}")
        print(f"Train Class: {train_cls_loss:.4f} | Val Class: {val_cls_loss:.4f}")
        
        # Write to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/train', train_dice_loss, epoch)
        writer.add_scalar('Dice/val', val_dice_loss, epoch)
        writer.add_scalar('focal/train', train_focal_loss, epoch)
        writer.add_scalar('focal/val', val_focal_loss, epoch)
        writer.add_scalar('Class/train', train_cls_loss, epoch)
        writer.add_scalar('Class/val', val_cls_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler': scaler.state_dict(),  # Save scaler state
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print("Saved new best model!")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler': scaler.state_dict(),  # Save scaler state
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Close TensorBoard writer
    writer.close()
    print("Training completed!")
    
    
# Add this to your training loop before loss calculation
def check_tensor_for_nan(tensor, name="", print_stats=True):
    """Check if tensor contains NaN values."""
    has_nan = torch.isnan(tensor).any().item()
    
    if has_nan or print_stats:
        print(f"\n--- {name} ---")
        print(f"Contains NaN: {has_nan}")
        if not has_nan and print_stats:
            print(f"Min: {tensor.min().item()}, Max: {tensor.max().item()}, Mean: {tensor.mean().item()}")
    
    return has_nan


def print_model_parameters(model):
    print("Model Parameters:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name}: {param.shape} ({num_params:,} parameters)")
    
    print(f"\nTotal Trainable Parameters: {total_params:,}")
    print(f"Parameter Size: {total_params * 4 / (1024 * 1024):.2f} MB (assuming float32)")
    print(f"Half Precision Size: {total_params * 2 / (1024 * 1024):.2f} MB (assuming float16)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lung Nodule Segmentation and Classification Model")
    
    # Data arguments
    parser.add_argument("--paths_file", type=str, default="paths_with_diagnosis.txt",
                        help="Path to file containing paths and diagnoses")
    
    # Model arguments
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of diagnosis classes (default: 4)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training (default: 1)")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of epochs to train (default: 5)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay (default: 1e-5)")
    parser.add_argument("--lambda_dice", type=float, default=1.0,
                        help="Weight for Dice loss (default: 1.0)")
    parser.add_argument("--lambda_focal", type=float, default=1.0,
                        help="Weight for focal loss (default: 1.0)")
    parser.add_argument("--lambda_cls", type=float, default=1.0,
                        help="Weight for classification loss (default: 1.0)")
    
    # Mixed precision and optimization arguments
    parser.add_argument("--use_amp", action="store_true", 
                        help="Use automatic mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--clip_gradients", action="store_true",
                        help="Use gradient clipping")
    parser.add_argument("--clip_value", type=float, default=1.0,
                        help="Gradient clipping value")
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of data loading workers (default: 1)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save outputs")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Interval to save model checkpoints (default: 10 epochs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    train_model(args)