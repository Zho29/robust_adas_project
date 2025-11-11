# WUNet+YOLO: Robust Object Detection in Adverse Weather

A high-performance object detection system combining Weather U-Net preprocessing with YOLOv8 for robust detection in adverse weather conditions for autonomous driving applications.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Visualizations](#visualizations)
- [Dataset](#dataset)
- [Citation](#citation)


---

## 🎯 Overview

**WUNet+YOLO** is a modular two-stage pipeline for robust object detection in adverse weather conditions:

1. **WUNet (Weather U-Net)**: Deep learning-based image preprocessing that restores weather-degraded images
2. **YOLOv8n**: Lightweight, fast object detector for real-time performance

This approach achieves **109 FPS** on NVIDIA RTX A6000, making it suitable for real-time autonomous driving applications even in challenging weather conditions like fog and rain.

### Why WUNet+YOLO?

-  **Ultra-fast**: 109 FPS (9.18ms latency)
-  **Efficient**: Only 110.6 GFLOPs
-  **Weather-robust**: Maintains 64-75% mAP across all conditions
-  **Modular**: Easy to update and improve individual components
-  **Real-time**: 3x faster than real-time requirements

---

## 🏆 Key Results

### Detection Accuracy

| Weather Condition | mAP@0.5 | mAP@0.5:0.95 | Performance |
|-------------------|---------|--------------|-------------|
| **Normal** | 74.75% | 49.46% | Excellent ✅ |
| **Heavy Fog** | 64.67% | 40.93% | Good ✅ |
| **Heavy Rain** | 73.24% | 48.20% | Excellent ✅ |

### Key Highlights

-  **109 FPS** - 3x faster than real-time (30 FPS)
-  **Weather Robust** - Minimal degradation in adverse conditions
-  **Efficient** - Only 110.6 GFLOPs computational cost
-  **Resilient to Rain** - Less than 2% accuracy drop
-  **Production Ready** - Suitable for embedded deployment

---

## ✨ Features

- **Weather Preprocessing**: Advanced U-Net architecture for weather degradation removal
- **Real-time Detection**: YOLOv8n for fast, accurate object detection
- **Weather Augmentation**: Realistic fog and rain effects using imgaug
- **Comprehensive Metrics**: FLOPs, latency, throughput, and detection accuracy
- **Rich Visualizations**: Publication-quality performance charts
- **Modular Pipeline**: Independent training and optimization of each component
- **GPU Optimized**: CUDA-accelerated with proper synchronization
- **Complete Documentation**: Detailed results and analysis

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 12.1+ (for GPU acceleration)
- NVIDIA GPU (tested on RTX A6000)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Zho29/robust_adas_project.git
cd robust_adas_project
```

2. **Create virtual environment:**
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download KITTI dataset:**
```bash
# Download KITTI object detection dataset
# Place in appropriate directory structure
```

---

## 📖 Usage

### 1. Run Evaluation

```bash
cd models
python evaluate_yolo_final.py
```

This will evaluate the WUNet+YOLO pipeline on:
- Normal weather conditions
- Heavy fog conditions
- Heavy rain conditions

### 2. Generate Visualizations

```bash
python visualize_comparison.py
```

Creates comprehensive performance visualizations including:
- mAP performance across weather conditions
- Latency and throughput metrics
- Complete performance dashboard

### 3. Convert Dataset to YOLO Format

```bash
python convert_kitti_to_yolo.py
```

Converts KITTI annotations to YOLO format for training.

### 4. Train WUNet

```bash
cd models
python train.py
```

### 5. Measure Performance Metrics

```bash
python measure_metrics.py
```

Measures FLOPs, latency, and throughput for the complete pipeline.

---

## 🏗️ Model Architecture

### WUNet+YOLO Pipeline

```
Input Image (Degraded)
    ↓
[WUNet Preprocessing]
- U-Net architecture
- 31.0M parameters
- 106.5 GFLOPs
- 9.11 ms latency
    ↓
Enhanced Image
    ↓
[YOLOv8n Detection]
- Lightweight detector
- 3.0M parameters
- 4.1 GFLOPs
- 0.07 ms latency
    ↓
Final Detections
```

### Component Breakdown

**WUNet (Weather U-Net):**
- **Purpose**: Image restoration and enhancement
- **Architecture**: U-Net with skip connections
- **Parameters**: 31.0M (91% of total)
- **FLOPs**: 106.5G (96% of total)
- **Latency**: 9.11ms (99.2% of total)

**YOLOv8n:**
- **Purpose**: Object detection
- **Architecture**: YOLOv8 nano variant
- **Parameters**: 3.0M (9% of total)
- **FLOPs**: 4.1G (4% of total)
- **Latency**: 0.07ms (0.8% of total)

---

## 📊 Performance Metrics

### Computational Efficiency

**Total Pipeline:**
- Parameters: 34.1M
- FLOPs: 110.6 GFLOPs
- Latency: 9.18 ms
- Throughput: 109.0 FPS

**Memory Footprint:**
- Model size: ~136 MB
- GPU memory: ~2 GB (inference)

### Real-Time Capability

WUNet+YOLO exceeds real-time requirements by **3x**:
-  **109 FPS** achieved
-  **30 FPS** required for real-time
-  **3.6x margin** for multi-camera setups

### Applications Enabled

- High-speed autonomous driving
- Multi-camera systems (can handle 3+ cameras simultaneously)
- Embedded platform deployment
- Real-time video processing
- Edge device inference

---

## 📁 Project Structure

```
robust_adas_project/
├── models/
│   ├── weather_unet.py                 # WUNet implementation
│   ├── yolo_data.py                    # YOLO data handling
│   ├── train.py                        # WUNet training script
│   ├── evaluate_yolo_final.py          # YOLO+WUNet evaluation
│   ├── measure_metrics.py              
│   ├── visualize_comparison.py         
│   ├── weather_augmenter.py            # Weather effects
│   ├── weather_database.py             # Weather dataset handling
│   ├── comprehensive_comparison_summary.txt  
│   └── comparison_*.png              
├── convert_kitti_to_yolo.py           
├── requirements.txt                     
├── .gitignore                          
└── README.md                           
```

---

## 🗂️ Dataset

**KITTI Object Detection Dataset**
- 7,481 training images
- 7,518 test images
- 758 validation images (used in evaluation)
- 3 object classes: Car, Pedestrian, Cyclist

**Weather Augmentation:**
- **Normal**: Baseline clear weather conditions
- **Heavy Fog**: Severe visibility reduction (imgaug FogAugmenter)
- **Heavy Rain**: Intense precipitation (imgaug RainAugmenter)

**Data Format:**
- Original: KITTI format (txt annotations)
- Converted: YOLO format (normalized coordinates)

---

## 🔬 Technical Details

### Evaluation Setup
- **Device**: NVIDIA RTX A6000 (48GB VRAM)
- **Framework**: PyTorch 2.4.1 + CUDA 12.1
- **YOLO Version**: Ultralytics YOLOv8n
- **Latency Measurement**: 100 runs with CUDA synchronization
- **FLOPs Measurement**: THoP library

### Model Checkpoints
- **WUNet**: `checkpoints/RGB_whole_best.pth` (epoch 199)
- **YOLO**: `runs/detect/yolov8n_clear_retrain2/weights/best.pt`

### Training Details
- **WUNet Training**: L1 loss, Adam optimizer, 200 epochs
- **YOLO Training**: Fine-tuned on KITTI dataset
- **Batch Size**: 8 (WUNet), 16 (YOLO)
- **Learning Rate**: 1e-4 (WUNet), auto (YOLO)

---

## 🌧️ Weather Robustness

### Performance Across Conditions

The system maintains strong performance across all weather scenarios:

**Normal Conditions:**
- mAP@0.5: 74.75%
- Baseline performance

**Heavy Fog:**
- mAP@0.5: 64.67%
- 13.5% degradation
- Still suitable for assisted driving

**Heavy Rain:**
- mAP@0.5: 73.24%
- Only 2.0% degradation
- Nearly baseline performance

### Why It Works

1. **WUNet Preprocessing**: Restores visibility before detection
2. **Weather-Aware Training**: Trained on diverse conditions
3. **Modular Design**: Each component optimized independently
4. **Robust Features**: YOLO trained on enhanced images

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{shahzad2024robustadasenhancingrobustness,
      title={Robust ADAS: Enhancing Robustness of Machine Learning-based Advanced Driver Assistance Systems for Adverse Weather}, 
      author={Muhammad Zaeem Shahzad and Muhammad Abdullah Hanif and Muhammad Shafique},
      year={2024},
      eprint={2407.02581},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.02581}, 

```



