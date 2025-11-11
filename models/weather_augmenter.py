from PIL import Image
import numpy as np
import imgaug.augmenters as iaa
from pathlib import Path
import os
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

class WeatherAugmenter:
    def __init__(self, random_seed=42):
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.fog_augmenters = {
            'low': iaa.Sequential([
                iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.1), add=(-20, 10)),
                iaa.GaussianBlur(sigma=(0.0, 1.0)),
                iaa.Fog()
            ]),
            'medium': iaa.Sequential([
                iaa.MultiplyAndAddToBrightness(mul=(0.7, 0.9), add=(-10, 20)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.03*255)),
                iaa.GaussianBlur(sigma=(1.0, 2.5)),
                iaa.Fog()
            ]),
            'high': iaa.Sequential([
                iaa.MultiplyAndAddToBrightness(mul=(0.6, 0.8), add=(20, 50)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
                iaa.GaussianBlur(sigma=(2.5, 4.0)),
                iaa.Fog()
            ])
        }

        self.rain_augmenters = {
            'low': iaa.Sequential([
                iaa.Rain(speed=(0.1, 0.25), drop_size=(0.01, 0.02)),
            ]),
            'medium': iaa.Sequential([
                iaa.Rain(speed=(0.25, 0.5), drop_size=(0.02, 0.03)),
                iaa.MotionBlur(k=7, angle=[-15, 15]),
                iaa.LinearContrast((0.8, 1.0))
            ]),
            'high': iaa.Sequential([
                iaa.Rain(speed=(0.5, 0.7), drop_size=(0.03, 0.04)),
                iaa.MotionBlur(k=13, angle=[-25, 25]),
                iaa.LinearContrast((0.7, 0.9))
            ])
        }

        self.snow_augmenters = {
            'low': iaa.Sequential([
                iaa.Snowflakes(flake_size=(0.1, 0.3), speed=(0.01, 0.04), density=(0.01, 0.05)),
                iaa.Multiply((1.0, 1.1), per_channel=0.2) 
            ]),
            'medium': iaa.Sequential([
                iaa.Snowflakes(flake_size=(0.2, 0.6), speed=(0.02, 0.05), density=(0.04, 0.1)),
                iaa.GaussianBlur(sigma=(0.5, 1.5)),
                iaa.LinearContrast((0.85, 1.0))
            ]),
            'high': iaa.Sequential([
                iaa.Snowflakes(flake_size=(0.4, 0.8), speed=(0.03, 0.06), density=(0.1, 0.25)),
                iaa.GaussianBlur(sigma=(1.5, 3.0)),
                iaa.LinearContrast((0.75, 0.9))
            ])
        }

    def augment_image(self, image_path, weather_type, intensity='low'):
        
        image = Image.open(image_path)
        image_array = np.array(image)

        if weather_type == 'fog':
            augmenter = self.fog_augmenters[intensity]
        elif weather_type == 'snow':
            augmenter = self.snow_augmenters[intensity]
        elif weather_type == 'rain':
            augmenter = self.rain_augmenters[intensity]
        else:
            raise ValueError(f"Weather type '{weather_type}' not found")
        
        augmented_array = augmenter(image=image_array)
        augmented_image = Image.fromarray(augmented_array)
        return augmented_image


    def create_augmented_dataset(self, input_dir, output_dir, test_size=0.1, max_images=None):
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
  
        train_dir = output_path / "train"
        test_dir = output_path / "test"
        val_sets_dir = output_path / "validation_sets"
        
        train_dir.mkdir(parents=True,exist_ok=True)
        test_dir.mkdir(parents=True,exist_ok=True)
        val_sets_dir.mkdir(parents=True,exist_ok=True)

        
        all_image_files = sorted(list(input_path.glob("*.png")))
        if max_images:
            all_image_files = all_image_files[:max_images]
        
        train_files, test_files = train_test_split(all_image_files, test_size=test_size, random_state=42)
        
        print(f"Dataset split: {len(train_files)} training images, {len(test_files)} testing images")
        
        weather_types = ['fog', 'rain', 'snow']
        intensities = ['low', 'medium', 'high']

        print("Generating training set...")
        for img_path in tqdm(train_files, desc="Generating training dataset"):
            base_name = img_path.stem
            original = Image.open(img_path)
            original.save(train_dir / f"{base_name}_clear.png")

            for weather_type in weather_types:
                for intensity in intensities:
                    try:
                        augmented = self.augment_image(img_path, weather_type, intensity)
                        filename = f"{base_name}_{weather_type}_{intensity}.png"
                        augmented.save(train_dir / filename)
                    except Exception as e:
                        print(f"Error processing {img_path} with {weather_type}_{intensity}: {e}")
                
        
        print("Generating test set...")
        for img_path in tqdm(test_files, desc="Generating test dataset"):
            base_name = img_path.stem
            original = Image.open(img_path)
            original.save(test_dir / f"{base_name}_clear.png")
            
            for weather_type in weather_types:
                for intensity in intensities:
                    try:
                        augmented = self.augment_image(img_path, weather_type, intensity)
                        filename = f"{base_name}_{weather_type}_{intensity}.png"
                        augmented.save(test_dir / filename)
                    except Exception as e:
                        print(f"Error processing {img_path} with {weather_type}_{intensity}: {e}")
        

        print("Creating validation sets...")
        normal_dir = val_sets_dir / "normal"
        normal_dir.mkdir(exist_ok=True)
        
        for img_path in test_files:
            base_name = img_path.stem
            original = Image.open(img_path)
            original.save(normal_dir / f"{base_name}_clear.png")
        

        for weather_type in weather_types:
            for intensity in intensities:
                val_dir = val_sets_dir / f"{weather_type}_{intensity}"
                val_dir.mkdir(exist_ok=True)
                
                for img_path in test_files:
                    base_name = img_path.stem
                    try:
                        augmented = self.augment_image(img_path, weather_type, intensity)
                        filename = f"{base_name}_{weather_type}_{intensity}.png"
                        augmented.save(val_dir / filename)
                    except Exception as e:
                        print(f"Error creating validation set {weather_type}_{intensity}: {e}")
        
        self._print_dataset_stats(train_dir, test_dir, val_sets_dir)

    def _print_dataset_stats(self, train_dir, test_dir, val_sets_dir):

        train_count = len(list(train_dir.glob("*.png")))
        test_count = len(list(test_dir.glob("*.png")))
        
        print(f"\n===== Dataset Statistics ======")
        print(f"Training images: {train_count}")
        print(f"Test images: {test_count}")
        print(f"Total augmented images: {train_count + test_count}")
        
        val_dirs = list(val_sets_dir.iterdir())
        print(f"Validation sets created: {len(val_dirs)}")
        for val_dir in val_dirs:
            if val_dir.is_dir():
                count = len(list(val_dir.glob("*.png")))
                print(f"  - {val_dir.name}: {count} images")



if __name__ == "__main__":
    augmenter = WeatherAugmenter(random_seed=42)
    augmenter.create_augmented_dataset(
        input_dir="data/raw/kitti/training/image_2",   
        output_dir="data/processed/weather_dataset_split",    
        test_size=0.1,  
        max_images=None 
    )