import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import time
import json
from tqdm import tqdm

# Import your models and dataset
from weather_unet import WeatherUNet
from weather_database import WeatherDataModule

class WUNetTrainer:
    def __init__(self, train_dir, test_dir, save_dir="checkpoints", log_dir="logs"):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.save_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Training config from paper
        self.config = {
            'epochs': 100,
            'learning_rate': 0.01,
            'batch_sizes': {'whole': 24, 'crops': 40},
            'save_every': 10,
            'validate_every': 5
        }
        
        # Setup data module
        self.data_module = WeatherDataModule(train_dir, test_dir, self.config['batch_sizes'])

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {self.device}")

    def train_variant(self, color_space='RGB', image_type='whole'):
        """Train a specific WUNet variant"""
        variant_name = f"{color_space}_{image_type}"
        print(f"\n{'='*50}")
        print(f"Training WUNet variant: {variant_name}")
        print(f"{'='*50}")
        
        # Setup model
        model = WeatherUNet(in_channels=3, out_channels=3).to(self.device)
        
        # Setup data loaders
        train_loader, test_loader = self.data_module.get_dataloaders(color_space, image_type)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        
        # Training metrics
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        # Training loop
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
            for weather_batch, clear_batch in train_pbar:
                weather_batch = weather_batch.to(self.device)
                clear_batch = clear_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(weather_batch)
                loss = criterion(outputs, clear_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Update progress bar
                train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if (epoch + 1) % self.config['validate_every'] == 0:
                model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Val]")
                    for weather_batch, clear_batch in val_pbar:
                        weather_batch = weather_batch.to(self.device)
                        clear_batch = clear_batch.to(self.device)
                        
                        outputs = model(weather_batch)
                        loss = criterion(outputs, clear_batch)
                        
                        val_loss += loss.item()
                        val_batches += 1
                        
                        val_pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})
                
                avg_val_loss = val_loss / val_batches
                val_losses.append(avg_val_loss)
                
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save_checkpoint(model, optimizer, epoch, avg_val_loss, variant_name, is_best=True)
            
            # Regular checkpoint saving
            if (epoch + 1) % self.config['save_every'] == 0:
                self._save_checkpoint(model, optimizer, epoch, avg_train_loss, variant_name, is_best=False)
        
        # Save final metrics
        metrics = {
            'variant': variant_name,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs': self.config['epochs']
        }
        
        metrics_path = self.save_dir / f"{variant_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Training completed for {variant_name}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return metrics

    def train_all_variants(self):
        """Train all 4 WUNet variants from the paper"""
        variants = [
            ('RGB', 'whole'),
            ('RGB', 'crops'), 
            ('HSV', 'whole'),
            ('HSV', 'crops')
        ]
        
        all_results = {}
        
        for color_space, image_type in variants:
            start_time = time.time()
            metrics = self.train_variant(color_space, image_type)
            training_time = time.time() - start_time
            
            variant_name = f"{color_space}_{image_type}"
            metrics['training_time_hours'] = training_time / 3600
            all_results[variant_name] = metrics
            
            print(f"{variant_name} training completed in {training_time/3600:.2f} hours")
        
        # Save comparison results
        comparison_path = self.save_dir / "all_variants_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print comparison
        self._print_comparison(all_results)
        
        return all_results

    def _save_checkpoint(self, model, optimizer, epoch, loss, variant_name, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'variant': variant_name
        }
        
        if is_best:
            checkpoint_path = self.save_dir / f"{variant_name}_best.pth"
        else:
            checkpoint_path = self.save_dir / f"{variant_name}_epoch_{epoch+1}.pth"
        
        torch.save(checkpoint, checkpoint_path)

    def _print_comparison(self, results):
        """Print comparison of all variants"""
        print(f"\n{'='*60}")
        print("WUNET VARIANTS COMPARISON (Paper Results)")
        print(f"{'='*60}")
        
        print(f"{'Variant':<15} {'Best Val Loss':<15} {'Training Time':<15}")
        print("-" * 50)
        
        for variant_name, metrics in results.items():
            val_loss = metrics['best_val_loss']
            training_time = metrics['training_time_hours']
            print(f"{variant_name:<15} {val_loss:<15.6f} {training_time:<15.2f}h")
        
        # Find best variant
        best_variant = min(results.items(), key=lambda x: x[1]['best_val_loss'])
        print(f"\nBest performing variant: {best_variant[0]} (Val Loss: {best_variant[1]['best_val_loss']:.6f})")

    def evaluate_variant(self, variant_name, checkpoint_path):
        """Evaluate a trained variant"""
        color_space, image_type = variant_name.split('_')
        
        # Load model
        model = WeatherUNet(in_channels=3, out_channels=3).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get test loader
        _, test_loader = self.data_module.get_dataloaders(color_space, image_type)
        
        # Evaluate
        criterion = nn.MSELoss()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for weather_batch, clear_batch in tqdm(test_loader, desc=f"Evaluating {variant_name}"):
                weather_batch = weather_batch.to(self.device)
                clear_batch = clear_batch.to(self.device)
                
                outputs = model(weather_batch)
                loss = criterion(outputs, clear_batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"{variant_name} evaluation MSE: {avg_loss:.6f}")
        
        return avg_loss

# Training script
if __name__ == "__main__":
    # Setup trainer
    trainer = WUNetTrainer(
        train_dir="../data/processed/weather_dataset_split/train",
        test_dir="../data/processed/weather_dataset_split/test"
    )
    
    # Train all variants (comment out specific ones if needed)
    print("Starting training of all 4 WUNet variants...")
    results = trainer.train_all_variants()

    # Train specific variant:
    # results = trainer.train_variant('RGB', 'whole')
    
    print("Training completed!")
