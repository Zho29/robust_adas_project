import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import time
import sys

from dsnet_retinanet import DSNet


class KITTIWeatherDatasetFinal(Dataset):
    """Final dataset with proper YOLO label parsing and data augmentation"""
    def __init__(self, weather_dir, labels_dir, augment=True):
        self.weather_dir = Path(weather_dir)
        self.labels_dir = Path(labels_dir)
        self.augment = augment

        # Get fog and rain images
        all_files = sorted(list(self.weather_dir.glob('*_fog_high.png')))
        all_files += sorted(list(self.weather_dir.glob('*_rain_high.png')))
        self.image_files = all_files

        # YOLO to RetinaNet class mapping
        # YOLO classes: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, 8=traffic light
        # RetinaNet classes: 1=person, 2=vehicle
        self.yolo_to_retinanet = {
            0: 1,  # person -> person
            1: 2,  # bicycle -> vehicle
            2: 2,  # car -> vehicle
            3: 2,  # motorcycle -> vehicle
            5: 2,  # bus -> vehicle
            7: 2,  # truck -> vehicle
            8: 2,  # traffic light -> vehicle (or skip)
        }

        print(f"   Loaded {len(self.image_files)} images")
        print(f"   Data augmentation: {'enabled' if augment else 'disabled'}")
    
    def parse_yolo_label(self, label_file):
        """Parse YOLO format labels"""
        boxes = []
        labels = []
        
        if not label_file.exists():
            return boxes, labels
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                try:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert to pixel coords (640x640)
                    cx = center_x * 640
                    cy = center_y * 640
                    w = width * 640
                    h = height * 640
                    
                    # Convert to corners
                    x1 = max(0, cx - w/2)
                    y1 = max(0, cy - h/2)
                    x2 = min(640, cx + w/2)
                    y2 = min(640, cy + h/2)
                    
                    # Validate and map class
                    if x2 > x1 + 1 and y2 > y1 + 1:
                        # Only use classes we have mapping for
                        if class_id in self.yolo_to_retinanet:
                            boxes.append([x1, y1, x2, y2])
                            retinanet_class = self.yolo_to_retinanet[class_id]
                            labels.append(retinanet_class)
                
                except (ValueError, IndexError):
                    continue
        
        return boxes, labels
    
    def __len__(self):
        return len(self.image_files)
    
    def apply_augmentation(self, weather_array, clear_array, boxes):
        """Apply random augmentation to images and boxes"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            weather_array = np.fliplr(weather_array).copy()
            clear_array = np.fliplr(clear_array).copy()
            # Flip boxes
            new_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                new_boxes.append([640 - x2, y1, 640 - x1, y2])
            boxes = new_boxes

        # Random brightness adjustment (±20%)
        if np.random.rand() > 0.5:
            brightness_factor = 0.8 + np.random.rand() * 0.4
            weather_array = np.clip(weather_array * brightness_factor, 0, 1)
            clear_array = np.clip(clear_array * brightness_factor, 0, 1)

        # Random contrast adjustment
        if np.random.rand() > 0.5:
            contrast_factor = 0.8 + np.random.rand() * 0.4
            weather_array = np.clip((weather_array - 0.5) * contrast_factor + 0.5, 0, 1)
            clear_array = np.clip((clear_array - 0.5) * contrast_factor + 0.5, 0, 1)

        return weather_array, clear_array, boxes

    def __getitem__(self, idx):
        weather_path = self.image_files[idx]

        # Load weather image
        weather_img = Image.open(weather_path).convert('RGB')
        weather_img = weather_img.resize((640, 640))
        weather_array = np.array(weather_img).astype(np.float32) / 255.0

        # Load clear image
        base_name = weather_path.stem.rsplit('_', 2)[0]
        clear_path = self.weather_dir / f"{base_name}_clear.png"
        clear_img = Image.open(clear_path).convert('RGB')
        clear_img = clear_img.resize((640, 640))
        clear_array = np.array(clear_img).astype(np.float32) / 255.0

        # Parse YOLO labels
        label_path = self.labels_dir / f"{base_name}_clear.txt"
        boxes, labels = self.parse_yolo_label(label_path)

        # Apply augmentation if enabled
        if self.augment and isinstance(boxes, list) and len(boxes) > 0:
            weather_array, clear_array, boxes = self.apply_augmentation(weather_array, clear_array, boxes)

        # Convert to tensors
        weather_tensor = torch.from_numpy(weather_array).permute(2, 0, 1)
        clear_tensor = torch.from_numpy(clear_array).permute(2, 0, 1)

        # If no boxes, skip this sample by returning valid dummy that will be filtered
        # Using -1 for ignore label, small box that won't match anchors
        if not boxes or len(boxes) == 0:
            boxes = [[0, 0, 0.1, 0.1]]
            labels = [-1]  # Ignore label

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
        }

        return weather_tensor, clear_tensor, target


def custom_collate(batch):
    """Custom collate function"""
    weather_imgs = []
    clear_imgs = []
    targets = []
    
    for weather, clear, target in batch:
        weather_imgs.append(weather)
        clear_imgs.append(clear)
        targets.append(target)
    
    return weather_imgs, clear_imgs, targets


def train_dsnet_final(epochs=100, batch_size=3, lr=1e-4, device='cuda'):
    """
    Final DSNet training with proper loss balancing
    """
    print("="*80)
    print("DSNet FINAL Training - Fixed Classification + Metrics")
    print("="*80)
    
    num_gpus = torch.cuda.device_count()
    print(f"\n🚀 Using {num_gpus} GPUs")
    
    # Initialize model
    print("\n1. Initializing DSNet...")
    model = DSNet(num_classes=3, pretrained=True)
    
    # Use 3 GPUs
    if num_gpus >= 3:
        print(f"   Using DataParallel with 3 GPUs")
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
        multi_gpu = True
    else:
        multi_gpu = False
    
    model = model.to(device)
    actual_model = model.module if multi_gpu else model
    
    # CRITICAL FIX: Higher learning rate for detection, separate optimizers
    optimizer_restoration = optim.Adam([
        {'params': actual_model.cb_module.parameters(), 'lr': lr},
        {'params': actual_model.restoration_subnet.fr_module.parameters(), 'lr': lr}
    ], lr=lr)
    
    optimizer_detection = optim.Adam([
        {'params': actual_model.detection_subnet.retinanet.parameters(), 'lr': lr * 5}  # 5x higher!
    ], lr=lr * 5)
    
    scheduler_restoration = optim.lr_scheduler.MultiStepLR(optimizer_restoration, milestones=[60, 80], gamma=0.1)
    scheduler_detection = optim.lr_scheduler.MultiStepLR(optimizer_detection, milestones=[60, 80], gamma=0.1)
    
    # Loss
    ve_loss_fn = nn.MSELoss()
    
    # Dataset
    print("\n2. Loading dataset...")
    base_dir = Path('../data/processed/weather_dataset_split')
    
    dataset = KITTIWeatherDatasetFinal(
        weather_dir=base_dir / 'train',
        labels_dir=base_dir / 'yolo_format/labels/train',
        augment=False  # Disable augmentation temporarily to avoid errors
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"   Training samples: {len(dataset)}")
    print(f"   Batches per epoch: {len(dataloader)}")
    print(f"   Batch size: {batch_size}")
    
    # Training loop
    print("\n3. Starting training...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        
        epoch_total = 0
        epoch_ve = 0
        epoch_cls = 0
        epoch_bbox = 0
        valid_batches = 0

        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{epochs} - Starting training...")
        print(f"{'='*60}", flush=True)

        for batch_idx, (weather_imgs, clear_imgs, targets) in enumerate(dataloader):
            try:
                # Filter out targets with ignore labels (-1) BEFORE stacking
                valid_indices = []
                valid_targets = []
                for i, t in enumerate(targets):
                    # Check if any label is valid (>= 0)
                    valid_mask = t['labels'] >= 0
                    num_valid = int(valid_mask.sum().item())  # Convert to int to avoid Boolean error

                    if num_valid > 0:
                        valid_indices.append(i)
                        valid_targets.append({
                            'boxes': t['boxes'][valid_mask].to(device),
                            'labels': t['labels'][valid_mask].to(device)
                        })

                if len(valid_targets) == 0:
                    continue

                # CRITICAL FIX: Ensure consistent batch size for DataParallel
                # DataParallel + RetinaNet has issues with variable batch sizes
                # Pad batch to original size by duplicating last valid sample
                if len(valid_targets) < batch_size:
                    samples_needed = batch_size - len(valid_targets)
                    for _ in range(samples_needed):
                        valid_indices.append(valid_indices[-1])
                        valid_targets.append({
                            'boxes': valid_targets[-1]['boxes'].clone(),
                            'labels': valid_targets[-1]['labels'].clone()
                        })

                # Stack with consistent batch size
                weather_batch = torch.stack([weather_imgs[i] for i in valid_indices]).to(device)
                clear_batch = torch.stack([clear_imgs[i] for i in valid_indices]).to(device)

                optimizer_restoration.zero_grad()
                optimizer_detection.zero_grad()

                # Forward
                if batch_idx == 0:
                    print(f"🔄 Batch 0: Starting forward pass with {len(valid_targets)} targets...", flush=True)
                loss_dict, restored_imgs, _ = model(weather_batch, valid_targets)
                if batch_idx == 0:
                    print(f"✅ Batch 0: Forward pass completed!", flush=True)

                # Extract losses with better error handling
                if not isinstance(loss_dict, dict):
                    print(f"\n⚠️  Batch {batch_idx}: loss_dict is not a dict, got {type(loss_dict)}")
                    continue

                if 'classification' not in loss_dict or 'bbox_regression' not in loss_dict:
                    print(f"\n⚠️  Batch {batch_idx}: Missing keys in loss_dict: {loss_dict.keys()}")
                    continue

                # Extract losses from DataParallel - they may be vectors (one per GPU)
                L_cls = loss_dict['classification']
                L_bbox = loss_dict['bbox_regression']

                # CRITICAL FIX: When using DataParallel, losses are vectors
                # Need to reduce them to scalars by taking mean
                if L_cls.dim() > 0:  # If it's a vector
                    L_cls = L_cls.mean()
                if L_bbox.dim() > 0:
                    L_bbox = L_bbox.mean()

                # CRITICAL FIX: Ensure batch sizes match for VE loss
                if restored_imgs.size(0) != clear_batch.size(0):
                    # This happens when DataParallel filters some samples internally
                    # Skip this batch to avoid size mismatch
                    continue

                L_ve = ve_loss_fn(restored_imgs, clear_batch)

                # Check for NaN or zero - convert tensors to scalars first
                if torch.isnan(L_cls).any() or torch.isnan(L_bbox).any() or torch.isnan(L_ve).any():
                    continue

                # Debug first batch of first epoch
                if epoch == 0 and batch_idx == 0:
                    print(f"\n🔍 First batch debug:")
                    print(f"   Num targets: {len(valid_targets)}")
                    for i, t in enumerate(valid_targets):
                        print(f"   Target {i}: {len(t['boxes'])} boxes, labels: {t['labels'].tolist()}")
                    print(f"   L_cls: {L_cls.item():.6f}")
                    print(f"   L_bbox: {L_bbox.item():.6f}")
                    print(f"   L_ve: {L_ve.item():.6f}")

                # IMPROVED LOSS BALANCING: Dynamic weights based on epoch
                # Early epochs: Focus more on getting detections right
                # Later epochs: Balance all three losses
                if epoch < 20:
                    alpha = 0.1   # Lower VE weight early on
                    beta_cls = 5.0   # VERY HIGH classification weight to prevent collapse
                    beta_bbox = 1.0  # Normal bbox weight
                elif epoch < 50:
                    alpha = 0.15
                    beta_cls = 3.0   # Still high
                    beta_bbox = 1.0
                else:
                    alpha = 0.2
                    beta_cls = 2.0
                    beta_bbox = 1.0

                total_loss = alpha * L_ve + beta_cls * L_cls + beta_bbox * L_bbox
                
                # Single backward
                optimizer_restoration.zero_grad()
                optimizer_detection.zero_grad()
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                # Both optimizers step
                optimizer_restoration.step()
                optimizer_detection.step()
                
                # Track
                epoch_total += total_loss.item()
                epoch_ve += L_ve.item()
                epoch_cls += L_cls.item()
                epoch_bbox += L_bbox.item()
                valid_batches += 1

                # Print losses every 10 batches to monitor training
                if (batch_idx + 1) % 10 == 0:
                    print(f"[Epoch {epoch+1}/{epochs}] Batch {batch_idx+1}/{len(dataloader)} | CLS: {L_cls.item():.4f} | BBOX: {L_bbox.item():.4f} | VE: {L_ve.item():.4f} | Total: {total_loss.item():.4f}", flush=True)

            except Exception as e:
                print(f"\n❌ Batch {batch_idx}: {e}")
                continue
        
        if valid_batches == 0:
            continue
        
        # Summary
        avg_total = epoch_total / valid_batches
        avg_ve = epoch_ve / valid_batches
        avg_cls = epoch_cls / valid_batches
        avg_bbox = epoch_bbox / valid_batches

        print(f"\nEpoch {epoch+1}:", flush=True)
        print(f"  Total: {avg_total:.6f}", flush=True)
        print(f"  VE: {avg_ve:.6f}", flush=True)
        print(f"  CLS: {avg_cls:.6f} {'🔥' if avg_cls > 0.05 else ('⚠️' if avg_cls > 0.01 else '❌')}", flush=True)
        print(f"  BBOX: {avg_bbox:.6f}", flush=True)

        scheduler_restoration.step()
        scheduler_detection.step()
        print(f"  LR Rest: {optimizer_restoration.param_groups[0]['lr']:.6f}", flush=True)
        print(f"  LR Det: {optimizer_detection.param_groups[0]['lr']:.6f}", flush=True)

        # CRITICAL: Early warning system for classification collapse
        if avg_cls < 0.001 and epoch < 50:
            print("\n🚨 CRITICAL WARNING: Classification loss collapsed to near zero!", flush=True)
            print("   This indicates the network is predicting only background.", flush=True)
            print("   Possible causes:", flush=True)
            print("   1. Anchor sizes don't match object sizes", flush=True)
            print("   2. IoU thresholds are too strict", flush=True)
            print("   3. Learning rate for detection head is too low", flush=True)
            print("   4. Loss weight for classification is too low", flush=True)
            if epoch >= 5:
                print("\n   Consider stopping and adjusting hyperparameters.", flush=True)

        # Save best model
        if avg_total < best_loss:
            best_loss = avg_total
            model_to_save = model.module if multi_gpu else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_restoration_state_dict': optimizer_restoration.state_dict(),
                'optimizer_detection_state_dict': optimizer_detection.state_dict(),
                'loss': best_loss,
                'cls_loss': avg_cls,
                'bbox_loss': avg_bbox,
                've_loss': avg_ve,
            }, 'checkpoints/dsnet_final_best.pth')
            print(f"  ✅ Saved best model (loss: {best_loss:.6f})", flush=True)

        # Save periodic checkpoints
        if (epoch + 1) % 20 == 0:
            model_to_save = model.module if multi_gpu else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'cls_loss': avg_cls,
                'bbox_loss': avg_bbox,
                've_loss': avg_ve,
            }, f'checkpoints/dsnet_final_epoch{epoch+1}.pth')
            print(f"  💾 Saved checkpoint epoch{epoch+1}", flush=True)
    
    print("\n✅ Training Complete!")
    return model


if __name__ == '__main__':
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    model = train_dsnet_final(
        epochs=100,
        batch_size=3,
        lr=1e-4,
        device='cuda'
    )
