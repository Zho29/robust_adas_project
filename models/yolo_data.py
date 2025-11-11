import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil
import json

class YOLOTrainer:
    def __init__(self, data_dir, output_dir="yolo_models"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # YOLO variants to train (as per paper)
        self.yolo_variants = ['yolov8n', 'yolov8s', 'yolov8m']
        
    def prepare_yolo_dataset(self):
        """Prepare KITTI dataset in YOLO format"""
        
        # Create YOLO dataset structure
        yolo_data_dir = self.data_dir / "yolo_format"
        yolo_data_dir.mkdir(exist_ok=True)
        
        # Create train/val directories
        (yolo_data_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_data_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (yolo_data_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_data_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Copy only clear images for training
        train_clear_images = list((self.data_dir / "train").glob("*_clear.png"))
        val_clear_images = list((self.data_dir / "test").glob("*_clear.png"))
        
        print(f"Preparing YOLO dataset...")
        print(f"Train images: {len(train_clear_images)}")
        print(f"Val images: {len(val_clear_images)}")
        
        # Copy training images
        for img_path in train_clear_images:
            dest_path = yolo_data_dir / "images" / "train" / img_path.name
            shutil.copy2(img_path, dest_path)
        
        # Copy validation images  
        for img_path in val_clear_images:
            dest_path = yolo_data_dir / "images" / "val" / img_path.name
            shutil.copy2(img_path, dest_path)
        
        # Create dataset config file
        self._create_dataset_config(yolo_data_dir)
        
        return yolo_data_dir
    
    def _create_dataset_config(self, yolo_data_dir):
        """Create YOLO dataset configuration file"""
        
        # KITTI classes (simplified for object detection)
        kitti_classes = [
            'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 
            'Cyclist', 'Tram', 'Misc', 'DontCare'
        ]
        
        dataset_config = {
            'path': str(yolo_data_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(kitti_classes)},
            'nc': len(kitti_classes)
        }
        
        config_path = yolo_data_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset config saved to: {config_path}")
        return config_path
    
    def train_yolo_variants(self):
        """Train all YOLO variants as per paper"""
        
        # Prepare dataset first
        yolo_data_dir = self.prepare_yolo_dataset()
        config_path = yolo_data_dir / "dataset.yaml"
        
        results = {}
        
        for variant in self.yolo_variants:
            print(f"\n{'='*50}")
            print(f"Training {variant} on clear KITTI images")
            print(f"{'='*50}")
            
            try:
                # Initialize YOLO model
                model = YOLO(f'{variant}.pt')
                
                # Training configuration (matching paper settings)
                train_results = model.train(
                    data=str(config_path),
                    epochs=100,
                    batch=16,
                    imgsz=640,
                    patience=10,
                    save=True,
                    project=str(self.output_dir),
                    name=f"{variant}_clear_baseline",workers=0,
                )
                
                # Save model
                model_path = self.output_dir / f"{variant}_clear_baseline" / "weights" / "best.pt"
                
                # Evaluate on validation set
                val_results = model.val(data=str(config_path))
                
                results[variant] = {
                    'model_path': str(model_path),
                    'mAP50': float(val_results.box.map50),
                    'mAP50_95': float(val_results.box.map),
                    'training_completed': True
                }
                
                print(f"{variant} training completed!")
                print(f"mAP@0.5: {results[variant]['mAP50']:.3f}")
                print(f"mAP@0.5:0.95: {results[variant]['mAP50_95']:.3f}")
                
            except Exception as e:
                print(f"Error training {variant}: {e}")
                results[variant] = {'error': str(e), 'training_completed': False}
        
        # Save results
        results_path = self.output_dir / "yolo_baseline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self._print_yolo_comparison(results)
        return results
    
    def _print_yolo_comparison(self, results):
        """Print comparison of YOLO variants"""
        print(f"\n{'='*60}")
        print("YOLO BASELINE PERFORMANCE (Clear Images Only)")
        print(f"{'='*60}")
        
        print(f"{'Variant':<15} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'Status':<15}")
        print("-" * 60)
        
        for variant, metrics in results.items():
            if metrics.get('training_completed', False):
                mAP50 = f"{metrics['mAP50']:.3f}"
                mAP50_95 = f"{metrics['mAP50_95']:.3f}"
                status = "Success"
            else:
                mAP50 = "Failed"
                mAP50_95 = "Failed" 
                status = "Error"
            
            print(f"{variant:<15} {mAP50:<12} {mAP50_95:<15} {status:<15}")
    
    def test_on_weather_conditions(self, model_variant='yolov8n'):
        """Test YOLO baseline on different weather conditions"""
        
        model_path = self.output_dir / f"{model_variant}_clear_baseline" / "weights" / "best.pt"
        
        if not model_path.exists():
            print(f"Model {model_path} not found. Train YOLO first.")
            return
        
        model = YOLO(str(model_path))
        
        # Test on different weather validation sets
        weather_conditions = ['normal', 'fog_high', 'rain_high', 'snow_high']
        results = {}
        
        for condition in weather_conditions:
            val_dir = self.data_dir / "validation_sets" / condition
            
            if val_dir.exists():
                print(f"\nTesting on {condition} conditions...")
                
                # Get sample images for testing
                test_images = list(val_dir.glob("*.png"))[:10]  # Test on subset
                
                if test_images:
                    # Run inference
                    test_results = model.predict(
                        source=[str(img) for img in test_images],
                        save=False,
                        verbose=False
                    )
                    
                    # Calculate average confidence
                    avg_confidence = 0
                    detection_count = 0
                    
                    for result in test_results:
                        if len(result.boxes) > 0:
                            avg_confidence += result.boxes.conf.mean().item()
                            detection_count += len(result.boxes)
                    
                    if len(test_results) > 0:
                        avg_confidence /= len(test_results)
                    
                    results[condition] = {
                        'avg_confidence': avg_confidence,
                        'detection_count': detection_count,
                        'images_tested': len(test_images)
                    }
                    
                    print(f"Average confidence: {avg_confidence:.3f}")
                    print(f"Total detections: {detection_count}")
        
        return results

# Usage
if __name__ == "__main__":
    trainer = YOLOTrainer(
        data_dir="../data/processed/weather_dataset_split"
    )
    
    # Train YOLO variants on clear images
    print("Training YOLO baselines on clear images...")
    yolo_results = trainer.train_yolo_variants()
    
    # Test performance degradation on weather conditions
    print("\nTesting baseline performance on weather conditions...")
    weather_results = trainer.test_on_weather_conditions('yolov8n')
    
    print("YOLO baseline training and testing completed!")
