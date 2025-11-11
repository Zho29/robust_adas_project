import torch
import time
import numpy as np
import sys
import os
from thop import profile, clever_format

# Import from existing files in the same directory
from dsnet_retinanet import DSNet
from weather_unet import WeatherUNet  # Direct import!


def load_real_wunet():
    """Load WUNet using existing architecture"""
    print("  📂 Loading WUNet...")
    model = WeatherUNet()  # Just use the class directly!
    
    checkpoint_paths = [
        '../wunet_implementation/wunet_RGB_whole_best.pth',
        '../wunet_implementation/checkpoints/wunet_RGB_whole_best.pth',
    ]
    
    for checkpoint_path in checkpoint_paths:
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  ✅ Loaded from: {checkpoint_path}")
                print(f"  ✅ Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 0):.6f}")
                return model
            except Exception as e:
                print(f"  ⚠️ Error: {e}")
    
    print("  ⚠️ No checkpoint found")
    return model


def load_yolo():
    """Load YOLOv8n"""
    print("  📂 Loading YOLOv8n...")
    try:
        from ultralytics import YOLO
        
        yolo_paths = [
            '../yolo_baseline/yolov8n.pt',
            '../yolo_baseline/runs/detect/train/weights/best.pt',
        ]
        
        for path in yolo_paths:
            if os.path.exists(path):
                yolo = YOLO(path)
                print(f"  ✅ Loaded from: {path}")
                return yolo.model
        
        yolo = YOLO('yolov8n.pt')
        print("  ✅ Loaded (default)")
        return yolo.model
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def load_dsnet():
    """Load DSNet"""
    print("  📂 Loading DSNet...")
    model = DSNet(num_classes=3, pretrained=False)
    
    checkpoint_paths = [
        'checkpoints/dsnet_final_best.pth',
        'checkpoints/dsnet_v2_best.pth',
        'checkpoints/dsnet_retinanet_best.pth'
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  ✅ Loaded from: {path}")
                print(f"  ✅ Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.6f}")
                if 'cls_loss' in checkpoint:
                    print(f"  ✅ CLS Loss: {checkpoint['cls_loss']:.6f}")
                return model
            except Exception as e:
                print(f"  ⚠️ Error: {e}")
    
    print("  ⚠️ No checkpoint found")
    return model


def measure_metrics(model, input_tensor, model_name, device='cuda', num_runs=100):
    """Measure all metrics"""
    print(f"\n{'='*70}")
    print(f"📊 {model_name}")
    print(f"{'='*70}")
    
    if model is None:
        return {'flops': 0, 'params': 0, 'latency_ms': 0, 'fps': 0}
    
    model = model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)
    
    # Parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params/1e6:.2f}M")
    
    # FLOPs
    try:
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        print(f"  FLOPs: {flops/1e9:.2f}G")
    except:
        flops = 2 * params * 640 * 640
        print(f"  FLOPs (est): {flops/1e9:.2f}G")
    
    # Latency
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            try:
                _ = model(input_tensor)
            except:
                pass
        
        # Measure
        latencies = []
        for _ in range(num_runs):
            try:
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                _ = model(input_tensor)
                if device == 'cuda':
                    torch.cuda.synchronize()
                latencies.append((time.time() - start) * 1000)
            except:
                break
    
    if latencies:
        avg_lat = np.mean(latencies)
        fps = 1000 / avg_lat
        print(f"  Latency: {avg_lat:.2f} ms")
        print(f"  FPS: {fps:.2f}")
    else:
        avg_lat = 0
        fps = 0
    
    return {'flops': flops, 'params': params, 'latency_ms': avg_lat, 'fps': fps}


def main():
    print("="*80)
    print("🔬 METRICS COMPARISON: WUNet vs DSNet")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    dummy = torch.randn(1, 3, 640, 640)
    results = {}
    
    # Measure each model
    print("\n" + "="*80)
    print("1️⃣  WUNet")
    print("="*80)
    results['WUNet'] = measure_metrics(load_real_wunet(), dummy, "WUNet", device)
    
    print("\n" + "="*80)
    print("2️⃣  YOLOv8n")
    print("="*80)
    results['YOLO'] = measure_metrics(load_yolo(), dummy, "YOLO", device)
    
    print("\n" + "="*80)
    print("3️⃣  DSNet")
    print("="*80)
    results['DSNet'] = measure_metrics(load_dsnet(), dummy, "DSNet", device)
    
    # Pipeline
    if results['WUNet']['flops'] > 0 and results['YOLO']['flops'] > 0:
        results['WUNet+YOLO'] = {
            'flops': results['WUNet']['flops'] + results['YOLO']['flops'],
            'params': results['WUNet']['params'] + results['YOLO']['params'],
            'latency_ms': results['WUNet']['latency_ms'] + results['YOLO']['latency_ms'],
            'fps': 1000 / (results['WUNet']['latency_ms'] + results['YOLO']['latency_ms']) 
                   if (results['WUNet']['latency_ms'] + results['YOLO']['latency_ms']) > 0 else 0
        }
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"{'Method':<20} {'FLOPs(G)':<12} {'Params(M)':<12} {'Latency(ms)':<15} {'FPS':<10}")
    print("-"*70)
    
    for name in ['WUNet', 'YOLO', 'WUNet+YOLO', 'DSNet']:
        if name in results and results[name]['flops'] > 0:
            r = results[name]
            print(f"{name:<20} {r['flops']/1e9:<12.2f} {r['params']/1e6:<12.2f} "
                  f"{r['latency_ms']:<15.2f} {r['fps']:<10.2f}")
    
    # Save
    import json
    os.makedirs('../results', exist_ok=True)
    with open('../results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved: results/metrics.json")


if __name__ == '__main__':
    main()