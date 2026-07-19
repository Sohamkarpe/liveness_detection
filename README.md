# Liveness Detection using Computer Vision

**Author:** Soham Ashok Karpe, M. Eng.  
**Institution:** Technische Hochschule Deggendorf (DIT), Campus Cham  
**Program:** M.Eng. Artificial Intelligence for Smart Sensors and Actuators  
**Period:** Mar 2024 – Jun 2024 | DIT Case study Project  

---

## 📌 Project Overview

A real-time human liveness detection system using **YOLOv8** across a **multi-camera setup**, achieving **91% detection accuracy**. The system distinguishes between live (real) persons and spoofing attempts (photos, videos, masks) using deep learning-based computer vision.

The complete inference pipeline is optimized for **edge platforms** (NVIDIA Jetson, Raspberry Pi) and validated under **varying lighting conditions** and **camera angles**.

---

## 🎯 Key Features

- ✅ Real-time liveness detection using YOLOv8
- ✅ Multi-camera synchronized capture and processing
- ✅ Edge-optimized inference pipeline (TensorRT support)
- ✅ Robust under varying lighting conditions and camera angles
- ✅ 91% detection accuracy achieved
- ✅ Configurable confidence thresholds
- ✅ Live visualization with bounding boxes and confidence scores

---

## 🏗️ Project Structure

```
liveness_detection/
├── src/
│   ├── main.py                    # Main entry point
│   ├── detector.py                # YOLOv8 liveness detector
│   ├── camera_manager.py          # Multi-camera management
│   ├── inference_pipeline.py      # Real-time inference pipeline
│   └── visualizer.py              # Live visualization
├── config/
│   └── config.yaml                # Configuration file
├── utils/
│   ├── preprocessing.py           # Image preprocessing
│   ├── postprocessing.py          # Detection postprocessing
│   └── logger.py                  # Logging utility
├── models/
│   └── .gitkeep                   # Place your trained model here
├── docs/
│   └── architecture.md            # System architecture description
├── requirements.txt
└── README.md
```

---

## ⚙️ System Architecture

```
Camera Input (Multi-Camera)
        ↓
Frame Synchronization
        ↓
Image Preprocessing
(resize, normalize, augment)
        ↓
YOLOv8 Inference
(liveness classification)
        ↓
Post-Processing
(NMS, confidence filtering)
        ↓
Result: LIVE / SPOOF
        ↓
Visualization + Logging
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning** | YOLOv8 (Ultralytics) |
| **Computer Vision** | OpenCV |
| **Programming** | Python 3.10+ |
| **Edge Deployment** | TensorRT, ONNX |
| **Hardware** | NVIDIA Jetson, Raspberry Pi |
| **Camera Interface** | OpenCV, GigE Vision |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Sohamkarpe/liveness_detection.git
cd liveness_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Run with default camera (webcam)
python src/main.py

# Run with specific camera index
python src/main.py --camera 0

# Run with multiple cameras
python src/main.py --cameras 0 1 2

# Run with custom config
python src/main.py --config config/config.yaml

# Run with custom model
python src/main.py --model models/liveness_model.pt

# Edge deployment mode (TensorRT)
python src/main.py --device cuda --trt
```

---

## 📊 Performance Results

| Metric | Value |
|--------|-------|
| **Detection Accuracy** | 91% |
| **Inference Speed (GPU)** | ~15ms/frame |
| **Inference Speed (CPU)** | ~85ms/frame |
| **Inference Speed (Jetson)** | ~35ms/frame |
| **FPS (real-time)** | 25-30 FPS |
| **Camera Setup** | Multi-camera |

---

## 📁 Dataset

> ⚠️ **Note:** The dataset used in this project is **not publicly available** due to privacy and institutional permissions. The dataset was collected and annotated internally at Technische Hochschule Deggendorf as part of a supervised research project.

**Dataset Characteristics:**
- Multi-class: Live person / Spoof (photo, video, mask)
- Multi-angle: Front, 45°, side view
- Varying lighting: Indoor, outdoor, low-light
- Multi-camera: Different focal lengths and resolutions

To use this project with your own dataset:
1. Prepare images in YOLO format
2. Update `config/config.yaml` with dataset path
3. Run training script

---

## 🏋️ Training (with your own dataset)

```bash
# Prepare dataset in YOLO format
# dataset/
#   images/train/  images/val/
#   labels/train/  labels/val/
#   data.yaml

# Train model
python src/train.py \
    --data dataset/data.yaml \
    --model yolov8m.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16
```

---

## 🔬 Methodology

### 1. Data Preprocessing
- Frame extraction from multi-camera feeds
- Image resizing to 640×640
- Normalization and augmentation (brightness, contrast, flip)

### 2. Model Architecture
- Base: YOLOv8 medium (YOLOv8m)
- Transfer learning from COCO pretrained weights
- Fine-tuned on liveness detection dataset
- Two-class output: `live` / `spoof`

### 3. Inference Pipeline
- Real-time frame capture from synchronized cameras
- Batch inference for efficiency
- Non-Maximum Suppression (NMS) post-processing
- Confidence threshold filtering (default: 0.5)

### 4. Edge Optimization
- TensorRT conversion for NVIDIA Jetson
- ONNX export for cross-platform deployment
- FP16 precision for faster inference

---

## 📝 Academic Context

This project was developed as part of the ** Case study ** program at **Technische Hochschule Deggendorf, Campus Cham**.

**Field:** Computer Vision, Deep Learning, Edge AI

---

## 📄 License

This project is for academic and research purposes. Please contact the author for commercial use.

---

## 👤 Author

**Soham Ashok Karpe, M. Eng.**  
AI Systems Engineer | Machine Vision & Deep Learning  
📧 karpesoham@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/soham-karpe)  
🐙 [GitHub](https://github.com/Sohamkarpe)
