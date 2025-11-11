import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import json
from PIL import Image
import numpy as np

from dsnet_retinanet import DSNet

def evaluate_dsnet_on_testset(checkpoint_path, test_image_dir, test_label_dir, 
                               weather_types=['fog_high', 'rain_high', 'clear'],
                               device='cuda'):
    """
    Evaluate DSNet on different weather conditions
    """
    print("="*80)
    print("DSNet Evaluation")
    print("="*80)
    
    # Load model
    print("\n1. Loading DSNet model...")
    model = DSNet(num_classes=3, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Training loss: {checkpoint['loss']:.6f}")
    
    results = {}
    
    # Evaluate on each weather condition
    for weather in weather_types:
        print(f"\n2. Evaluating on {weather} images...")
        
        # Get images for this weather type
        if weather == 'clear':
            image_pattern = f"*_clear.png"
        else:
            image_pattern = f"*_{weather}.png"
        
        image_dir = Path(test_image_dir)
        image_files = sorted(list(image_dir.glob(image_pattern)))
        
        print(f"   Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print(f"   ⚠️  No images found, skipping {weather}")
            continue
        
        # Process images
        total_detections = 0
        total_gt_boxes = 0
        
        all_predictions = []
        all_ground_truths = []
        
        for img_path in tqdm(image_files, desc=f"Processing {weather}"):
            # Load image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((640, 640))
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Run detection
            with torch.no_grad():
                detections, restored_img = model(img_tensor)
            
            # Count detections
            if len(detections) > 0:
                boxes = detections[0]['boxes']
                scores = detections[0]['scores']
                labels = detections[0]['labels']
                
                # Filter by confidence > 0.5
                high_conf = scores > 0.5
                num_dets = high_conf.sum().item()
                total_detections += num_dets
                
                all_predictions.append({
                    'image': img_path.name,
                    'boxes': boxes[high_conf].cpu().tolist(),
                    'scores': scores[high_conf].cpu().tolist(),
                    'labels': labels[high_conf].cpu().tolist()
                })
            
            # Load ground truth
            base_name = img_path.stem.rsplit('_', 2)[0] if weather != 'clear' else img_path.stem.replace('_clear', '')
            label_path = Path(test_label_dir) / f"{base_name}.txt"
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    gt_boxes = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 8:
                            class_name = parts[0]
                            if class_name in ['Pedestrian', 'Car']:
                                gt_boxes.append(parts[4:8])
                    total_gt_boxes += len(gt_boxes)
                    
                    all_ground_truths.append({
                        'image': img_path.name,
                        'boxes': gt_boxes,
                        'class': class_name
                    })
        
        # Summary for this weather type
        avg_detections = total_detections / len(image_files) if len(image_files) > 0 else 0
        avg_gt = total_gt_boxes / len(image_files) if len(image_files) > 0 else 0
        
        results[weather] = {
            'num_images': len(image_files),
            'total_detections': total_detections,
            'avg_detections_per_image': avg_detections,
            'total_ground_truth': total_gt_boxes,
            'avg_gt_per_image': avg_gt,
            'predictions': all_predictions,
            'ground_truths': all_ground_truths
        }
        
        print(f"\n   Results for {weather}:")
        print(f"   - Total detections: {total_detections}")
        print(f"   - Avg detections/image: {avg_detections:.2f}")
        print(f"   - Total ground truth boxes: {total_gt_boxes}")
        print(f"   - Avg GT boxes/image: {avg_gt:.2f}")
    
    return results


def save_results(results, output_file='dsnet_evaluation_results.json'):
    """Save evaluation results to JSON"""
    # Remove detailed predictions/GT for summary
    summary = {}
    for weather, data in results.items():
        summary[weather] = {
            'num_images': data['num_images'],
            'total_detections': data['total_detections'],
            'avg_detections_per_image': data['avg_detections_per_image'],
            'total_ground_truth': data['total_ground_truth'],
            'avg_gt_per_image': data['avg_gt_per_image']
        }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")


if __name__ == '__main__':
    # Configuration
    CHECKPOINT = 'checkpoints/dsnet_retinanet_best.pth'
    
    # Use validation set from weather_dataset_split
    TEST_IMAGE_DIR = '../data/processed/weather_dataset_split/validation_sets/extreme_weather/images'
    TEST_LABEL_DIR = '../data/processed/weather_dataset_split/yolo_format/labels/val'
    
    # Check if extreme_weather validation exists, else use train images
    import os
    if not os.path.exists(TEST_IMAGE_DIR):
        print("Using train directory for evaluation...")
        TEST_IMAGE_DIR = '../data/processed/weather_dataset_split/train'
    
    WEATHER_TYPES = ['fog_high', 'rain_high', 'clear']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Test images: {TEST_IMAGE_DIR}")
    print(f"Test labels: {TEST_LABEL_DIR}")
    print(f"Device: {DEVICE}\n")
    
    # Evaluate
    results = evaluate_dsnet_on_testset(
        checkpoint_path=CHECKPOINT,
        test_image_dir=TEST_IMAGE_DIR,
        test_label_dir=TEST_LABEL_DIR,
        weather_types=WEATHER_TYPES,
        device=DEVICE
    )
    
    # Save results
    save_results(results, 'dsnet_evaluation_results.json')
    
    print("\n" + "="*80)
    print("✅ Evaluation Complete!")
    print("="*80)
