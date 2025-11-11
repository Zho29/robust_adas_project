import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
from torchvision import transforms


class WeatherDataset(Dataset):
    
    def __init__(self,data_dir,color_space='RGB'):
        self.data_dir = Path(data_dir)
        self.color_space = color_space
        
        self.clear_images = sorted(list(self.data_dir.glob("*_clear.png")))
    
        self.image_pairs = []
        self._create_pairs()
        
        self.transform = transforms.Compose([
            transforms.Resize((200, 640)),
            transforms.ToTensor()
        ])


    def _create_pairs(self):
        weather_types = ['fog', 'rain', 'snow']
        intensities = ['low', 'medium', 'high']
        
        for clear_path in self.clear_images:
            base_name = clear_path.stem.replace('_clear', '')
            
            for weather in weather_types:
                for intensity in intensities:
                    weather_file = f"{base_name}_{weather}_{intensity}.png"
                    weather_path = self.data_dir / weather_file
                    
                    if weather_path.exists():
                        self.image_pairs.append((weather_path, clear_path))
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        weather_path, clear_path = self.image_pairs[idx]
        
        weather_img = Image.open(weather_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')
        
        if self.color_space == 'HSV':
            weather_img = self._rgb_to_hsv(weather_img)
            clear_img = self._rgb_to_hsv(clear_img)
        

        weather_tensor = self.transform(weather_img)
        clear_tensor = self.transform(clear_img)
        
        return weather_tensor, clear_tensor
    
    def _rgb_to_hsv(self, img):
        img_array = np.array(img)
        hsv_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        return Image.fromarray(hsv_array)

class WeatherDataModule:
    def __init__(self, train_dir, test_dir, batch_sizes):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_sizes = batch_sizes
    
    def get_dataloaders(self, color_space='RGB', image_type='whole'):
        batch_size = self.batch_sizes[image_type]
        
        train_dataset = WeatherDataset(self.train_dir, color_space)
        test_dataset = WeatherDataset(self.test_dir, color_space)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        return train_loader, test_loader