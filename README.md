# Robust ADAS: Object Detection in Adverse Weather Conditions

A comprehensive evaluation and comparison of deep learning approaches for robust object detection in adverse weather conditions, specifically comparing **DSNet** and **WUNet+YOLO** architectures for autonomous driving applications.

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
- [Model Architectures](#model-architectures)
- [Evaluation Results](#evaluation-results)
- [Visualizations](#visualizations)
- [Dataset](#dataset)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project provides a comprehensive comparison between two state-of-the-art approaches for robust object detection in adverse weather conditions:

1. **DSNet (Dual-Subnet Network)**: End-to-end architecture combining detection and restoration subnets
2. **WUNet+YOLO**: Modular pipeline with Weather U-Net preprocessing followed by YOLOv8n detection

The evaluation focuses on three critical metrics:
- **mAP@0.5**: Detection accuracy
- **FLOPs**: Computational complexity
- **Latency**: Real-time inference speed

---

## 🏆 Key Results

### Winner: WUNet+YOLO

| Metric | DSNet | WUNet+YOLO | Advantage |
|--------|-------|------------|-----------|
| **Latency** | 27.60 ms | 9.18 ms | **3.01x faster** ⚡ |
| **FLOPs** | 191.3 G | 110.6 G | **42.2% reduction** 🎯 |
| **FPS** | 36.2 | 109.0 | **3x throughput** 🚀 |
| **Parameters** | 32.4 M | 34.1 M | +5.0% |

### Detection Performance (WUNet+YOLO)

| Weather Condition | mAP@0.5 | mAP@0.5:0.95 | Degradation |
|-------------------|---------|--------------|-------------|
| Normal | 74.75% | 49.46% | - |
| Heavy Fog | 64.67% | 40.93% | -10.08% |
| Heavy Rain | 73.24% | 48.20% | -1.51% |

**Key Findings:**
- ✅ WUNet+YOLO is **3.01x faster** with 42.2% fewer FLOPs
- ✅ Both models achieve **real-time performance** (>30 FPS)
- ✅ Strong **weather robustness** across all conditions
- ✅ Particularly resilient to rain (minimal accuracy loss)

---

## ✨ Features

- **Comprehensive Evaluation Pipeline**: Automated comparison of multiple models
- **Weather Augmentation**: Realistic fog and rain effects using imgaug
- **Performance Metrics**: FLOPs, latency, throughput, and detection accuracy
- **Rich Visualizations**: Publication-quality charts and graphs
- **Modular Design**: Easy to extend with new models and weather conditions
- **GPU Accelerated**: Optimized for CUDA with proper synchronization
- **Detailed Documentation**: Complete results and analysis

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

### 1. Run Complete Comparison

```bash
cd models
python compare_dsnet_wunet.py
```

This will:
- Load DSNet, WUNet, and YOLO models
- Measure FLOPs and latency
- Evaluate mAP@0.5 on normal, fog, and rain conditions
- Save results to `dsnet_wunet_comparison_results.json`

### 2. Generate Visualizations

```bash
python visualize_comparison.py
```

Generates 4 comprehensive figures:
- Computational efficiency comparison
- mAP performance across weather conditions
- Speedup analysis
- Complete summary dashboard

### 3. Convert Dataset to YOLO Format

```bash
python convert_kitti_to_yolo.py
```

---

## 🏗️ Model Architectures

### DSNet (Dual-Subnet Network)

```
Input Image → [Detection Subnet] → Detections
           ↘ [Restoration Subnet] ↗
```

- **Detection Subnet**: RetinaNet-based architecture
- **Restoration Subnet**: Weather-aware image enhancement
- **Joint Training**: End-to-end learning
- **Parameters**: 32.4M

### WUNet+YOLO Pipeline

```
Input Image → [WUNet] → Enhanced Image → [YOLOv8n] → Detections
```

- **WUNet**: Weather U-Net for image preprocessing (31.0M params)
- **YOLOv8n**: Lightweight object detector (3.0M params)
- **Modular Design**: Independent training and updates
- **Total Parameters**: 34.1M

---

## 📊 Evaluation Results

### Computational Efficiency

**DSNet:**
- Parameters: 32.4M
- FLOPs: 191.3 GFLOPs
- Latency: 27.60 ms
- FPS: 36.2

**WUNet+YOLO:**
- Parameters: 34.1M (WUNet: 31.0M, YOLO: 3.0M)
- FLOPs: 110.6 GFLOPs (WUNet: 106.5G, YOLO: 4.1G)
- Latency: 9.18 ms (WUNet: 9.11ms, YOLO: 0.07ms)
- FPS: 109.0

**Speedup Analysis:**
- Latency: 3.01x faster
- FPS: 3.01x higher throughput
- FLOPs: 42.2% reduction
- Time saved: 18.42 ms per frame

### Real-Time Capability

Both models exceed the 30 FPS threshold for real-time operation:
- ✅ DSNet: 36.2 FPS (real-time capable)
- ✅ WUNet+YOLO: 109.0 FPS (3x real-time capable)

WUNet+YOLO's 109 FPS makes it suitable for:
- High-speed autonomous driving
- Embedded systems deployment
- Multi-camera setups
- Real-time video processing

---

## 📈 Visualizations

### 1. Computational Efficiency
![Computational Efficiency](models/comparison_computational_efficiency.png)

Comparison of parameters, FLOPs, latency, and FPS between DSNet and WUNet+YOLO.

### 2. mAP Performance
![mAP Performance](models/comparison_map_performance.png)

Detection accuracy across normal, fog, and rain conditions.

### 3. Speedup Analysis
![Speedup Analysis](models/comparison_speedup_analysis.png)

Detailed breakdown of latency distribution and speedup factors.

### 4. Summary Dashboard
![Summary Dashboard](models/comparison_summary_dashboard.png)

Complete performance overview with all key metrics.

---

## 📁 Project Structure

```
robust_adas_project/
├── models/
│   ├── compare_dsnet_wunet.py          # Main comparison script
│   ├── visualize_comparison.py          # Visualization generator
│   ├── dsnet_retinanet.py              # DSNet implementation
│   ├── weather_unet.py                  # WUNet implementation
│   ├── train_dsnet_final.py            # DSNet training
│   ├── evaluate_dsnet.py               # DSNet evaluation
│   ├── evaluate_yolo_final.py          # YOLO evaluation
│   ├── weather_augmenter.py            # Weather effects
│   ├── measure_metrics.py              # Metrics measurement
│   ├── yolo_data.py                    # YOLO data handling
│   ├── dsnet_wunet_comparison_results.json  # Results
│   ├── comprehensive_comparison_summary.txt  # Text summary
│   ├── COMPARISON_README.md            # Detailed documentation
│   └── comparison_*.png                # Visualizations
├── convert_kitti_to_yolo.py            # Dataset conversion
├── requirements.txt                     # Python dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

---

## 🗂️ Dataset

**KITTI Object Detection Dataset**
- 7,481 training images
- 7,518 test images
- 758 validation images (used in this evaluation)
- 3 object classes: Car, Pedestrian, Cyclist

**Weather Augmentation:**
- Normal conditions (baseline)
- Heavy fog (imgaug FogAugmenter)
- Heavy rain (imgaug RainAugmenter)

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
- **DSNet**: `checkpoints/dsnet_final_best.pth` (epoch 94)
- **WUNet**: `checkpoints/RGB_whole_best.pth` (epoch 199)
- **YOLO**: `runs/detect/yolov8n_clear_retrain2/weights/best.pt`

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{robust_adas_comparison,
  title={Robust ADAS: Comprehensive Comparison of DSNet and WUNet+YOLO for Object Detection in Adverse Weather},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/Zho29/robust_adas_project}
}
```

---

## 🙏 Acknowledgments

- **KITTI Dataset**: Vision benchmark suite for autonomous driving
- **Ultralytics**: YOLOv8 implementation and framework
- **imgaug**: Weather augmentation library
- **THoP**: FLOPs calculation tool

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **as19197** - Initial work and evaluation

---

## 📫 Contact

For questions or collaboration opportunities:
- GitHub: [@Zho29](https://github.com/Zho29)
- Project Link: [https://github.com/Zho29/robust_adas_project](https://github.com/Zho29/robust_adas_project)

---

## 🔄 Updates

**Last Updated**: November 2025

**Status**: ✅ Complete and Validated

---

## 🎯 Future Work

- [ ] Extend evaluation to more weather conditions (snow, night)
- [ ] Add more detection models (Faster R-CNN, DETR)
- [ ] Implement real-time video processing demo
- [ ] Deploy models on embedded platforms (Jetson)
- [ ] Add temporal consistency metrics for video
- [ ] Create web interface for interactive comparison

---

**Made with ❤️ for safer autonomous driving**
