import torch
from ultralytics import YOLO
from weather_unet import WeatherUNet
from pathlib import Path
import json
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa

class YOLOWeatherEvaluator:
    def __init__(self, yolo_model_path, wunet_model_path, device='cuda'):
        self.device = device
        
        # Load YOLO
        print(f"Loading YOLO from {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)
        
        # Load WUNet
        print(f"Loading WUNet from {wunet_model_path}")
        self.wunet = WeatherUNet(in_channels=3, out_channels=3).to(device)
        checkpoint = torch.load(wunet_model_path, map_location=device, weights_only=False)
        self.wunet.load_state_dict(checkpoint['model_state_dict'])
        self.wunet.eval()
        
        # Weather augmenters
        self.augmenters = {
            'fog_high': iaa.Sequential([
                iaa.MultiplyAndAddToBrightness(mul=(0.6, 0.8), add=(20, 50)),
                iaa.GaussianBlur(sigma=(2.5, 4.0)),
                iaa.Fog()
            ]),
            'rain_high': iaa.Sequential([
                iaa.Rain(speed=(0.5, 0.7), drop_size=(0.03, 0.04)),
                iaa.MotionBlur(k=13, angle=[-25, 25]),
            ]),
        }
        
        print("Models loaded successfully!")
    
    def preprocess_with_wunet(self, img_pil):
        """Apply WUNet to remove weather artifacts"""
        img = img_pil.resize((640, 200))
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            clear_tensor = self.wunet(img_tensor)
        
        clear_img = clear_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        clear_img = (clear_img * 255).astype(np.uint8)
        return Image.fromarray(clear_img)
    
    def evaluate_condition(self, weather_type, use_wunet=False):
        """Evaluate on a specific weather condition"""
        print(f"\n{'='*60}")
        print(f"{weather_type.upper()} | WUNet: {use_wunet}")
        print(f"{'='*60}")
        
        # Create temp dataset
        temp_dir = Path(f"temp_{weather_type}_wunet{use_wunet}")
        temp_images = temp_dir / "images" / "val"
        temp_labels = temp_dir / "labels" / "val"
        temp_images.mkdir(parents=True, exist_ok=True)
        temp_labels.mkdir(parents=True, exist_ok=True)
        
        # Source paths
        orig_images = Path("../data/processed/weather_dataset_split/yolo_format/images/val")
        orig_labels = Path("../data/processed/weather_dataset_split/yolo_format/labels/val")
        
        image_files = sorted(list(orig_images.glob("*_clear.png")))
        
        print(f"Processing {len(image_files)} images...")
        
        for img_path in tqdm(image_files):
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Apply weather if not normal
            if weather_type != 'normal':
                img_array = np.array(img)
                augmenter = self.augmenters[weather_type]
                img_array = augmenter(image=img_array)
                img = Image.fromarray(img_array)
            
            # Apply WUNet if requested
            if use_wunet:
                img = self.preprocess_with_wunet(img)
            
            # Save with same filename
            img.save(temp_images / img_path.name)
            
            # Copy label
            label_path = orig_labels / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, temp_labels / label_path.name)
        
        # Create YAML
        yaml_content = f"""path: {temp_dir.absolute()}
train: images/val
val: images/val
names:
  0: Car
  1: Van
  2: Truck
  3: Pedestrian
  4: Person_sitting
  5: Cyclist
  6: Tram
  7: Misc
  8: DontCare
nc: 9
"""
        yaml_path = temp_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        # Run validation
        print("Running YOLO validation...")
        results = self.yolo.val(data=str(yaml_path), verbose=False)
        
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50_95': float(results.box.map)
        }
        
        print(f"mAP@0.5: {metrics['mAP50']:.3f} | mAP@0.5:0.95: {metrics['mAP50_95']:.3f}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return metrics
    
    def quick_evaluation(self):
        """Quick evaluation on key conditions"""
        conditions = ['normal', 'fog_high', 'rain_high']
        
        results = {}
        
        for condition in conditions:
            baseline = self.evaluate_condition(condition, use_wunet=False)
            wunet = self.evaluate_condition(condition, use_wunet=True)
            
            results[condition] = {
                'baseline': baseline,
                'with_wunet': wunet,
                'improvement_mAP50': wunet['mAP50'] - baseline['mAP50']
            }
        
        # Save and print summary
        with open('yolo_wunet_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*70)
        print("SUMMARY: Impact of WUNet on YOLO Performance")
        print("="*70)
        print(f"{'Condition':<15} {'Baseline':<15} {'With WUNet':<15} {'Improvement':<15}")
        print("-"*70)
        
        for condition, data in results.items():
            baseline = data['baseline']['mAP50']
            wunet = data['with_wunet']['mAP50']
            improvement = data['improvement_mAP50']
            print(f"{condition:<15} {baseline:.3f} ({baseline*100:.1f}%)  {wunet:.3f} ({wunet*100:.1f}%)  {improvement:+.3f} ({improvement*100:+.1f}%)")
        
        return results

if __name__ == "__main__":
    yolo_model = "yolo_models/yolov8n_clear_baseline2/weights/best.pt"
    wunet_model = "checkpoints/RGB_whole_best.pth"
    
    evaluator = YOLOWeatherEvaluator(yolo_model, wunet_model, device='cuda')
    results = evaluator.quick_evaluation()
    
    print("\n Results saved to yolo_wunet_results.json")
