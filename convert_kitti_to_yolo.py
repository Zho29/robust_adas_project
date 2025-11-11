import os
from pathlib import Path

# KITTI class mapping to YOLO indices
KITTI_CLASSES = {
    'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
    'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6,
    'Misc': 7, 'DontCare': 8
}

def convert_kitti_to_yolo(kitti_label_path, yolo_label_path, img_width=1242, img_height=375):
    """Convert KITTI format to YOLO format"""
    
    with open(kitti_label_path, 'r') as f:
        lines = f.readlines()
    
    yolo_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15:
            continue
        
        class_name = parts[0]
        if class_name not in KITTI_CLASSES:
            continue
        
        class_id = KITTI_CLASSES[class_name]
        
        # KITTI bbox: left, top, right, bottom
        left = float(parts[4])
        top = float(parts[5])
        right = float(parts[6])
        bottom = float(parts[7])
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = ((left + right) / 2) / img_width
        y_center = ((top + bottom) / 2) / img_height
        width = (right - left) / img_width
        height = (bottom - top) / img_height
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Write YOLO format labels
    with open(yolo_label_path, 'w') as f:
        f.writelines(yolo_lines)

# Convert all labels
kitti_label_dir = Path("data/raw/kitti/training/label_2")
yolo_train_labels = Path("data/processed/weather_dataset_split/yolo_format/labels/train")
yolo_val_labels = Path("data/processed/weather_dataset_split/yolo_format/labels/val")

yolo_train_labels.mkdir(parents=True, exist_ok=True)
yolo_val_labels.mkdir(parents=True, exist_ok=True)

# Get list of images in train and val
train_images = list(Path("data/processed/weather_dataset_split/yolo_format/images/train").glob("*.png"))
val_images = list(Path("data/processed/weather_dataset_split/yolo_format/images/val").glob("*.png"))

print(f"Converting {len(train_images)} training labels...")
for img_path in train_images:
    img_id = img_path.stem.replace("_clear", "")
    kitti_label = kitti_label_dir / f"{img_id}.txt"
    yolo_label = yolo_train_labels / f"{img_path.stem}.txt"
    
    if kitti_label.exists():
        convert_kitti_to_yolo(kitti_label, yolo_label)

print(f"Converting {len(val_images)} validation labels...")
for img_path in val_images:
    img_id = img_path.stem.replace("_clear", "")
    kitti_label = kitti_label_dir / f"{img_id}.txt"
    yolo_label = yolo_val_labels / f"{img_path.stem}.txt"
    
    if kitti_label.exists():
        convert_kitti_to_yolo(kitti_label, yolo_label)

print("Conversion complete!")

