# Anime YOLO AI

Production-grade anime character detection system using YOLOv8 with full MLOps pipeline.

[![CI/CD](https://github.com/username/Anime-YOLO-AI/workflows/ML%20Pipeline%20CI/CD/badge.svg)](https://github.com/username/Anime-YOLO-AI/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

## 🎯 Project Overview

A Level 3+ ML system for real-time detection and classification of 5 anime characters:
- **Classes**: Naruto, Luffy, Gojo, Goku, Sukuna
- **Dataset**: 24,511 SafeBooru images (auto-downloaded and filtered)
- **Architecture**: YOLOv8n transfer learning from COCO-pretrained weights
- **Performance Tracking**: MLflow experiment logging, structured API monitoring
- **Deployment**: FastAPI REST API with Docker containerization

## 📁 Folder Structure
```
Anime-YOLO-AI/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline (tests, Docker build, nightly training)
├── api/
│   ├── main.py                 # FastAPI production endpoint with logging
│   ├── requirements.txt        # API dependencies
│   └── Dockerfile              # Container deployment
├── data/
│   └── raw/
│       ├── images/
│       │   ├── train/          # 17,280 training images
│       │   └── val/            # 4,321 validation images
│       └── labels/
│           ├── train/          # YOLO format labels
│           └── val/
├── runs/
│   ├── detect/
│   │   └── train/              # Training outputs (weights, metrics, plots)
│   ├── benchmark/              # EDA and baseline results
│   └── batch_preds/            # Batch inference outputs
├── src/
│   ├── download_and_filter.py  # SafeBooru dataset download pipeline
│   ├── validate_images.py      # Image corruption detection
│   ├── autolabel_faces.py      # Haar cascade auto-labeling
│   ├── split_dataset.py        # 80/20 stratified split
│   ├── augment.py              # Albumentations augmentation
│   ├── benchmark.py            # EDA + logistic regression baseline
│   ├── train.py                # YOLOv8 training script
│   ├── train_mlflow.py         # MLflow experiment tracking wrapper
│   ├── evaluate.py             # Comprehensive metrics (mAP, precision, recall)
│   ├── batch_predict.py        # Batch inference for nightly runs
│   ├── load_test.py            # Async API load testing (100 req, 10 concurrent)
│   ├── inference.py            # Real-time webcam/video inference
│   ├── track.py                # ByteTrack multi-object tracking
│   └── utils.py                # Shared utilities
├── all_data.csv                # SafeBooru metadata (200MB+)
├── manifest_filtered.csv       # Filtered dataset manifest (25,009 rows)
├── dataset.yaml                # YOLOv8 training configuration
├── requirements.txt            # Python dependencies
└── README.md
```

## 📊 Dataset

### SafeBooru Anime Characters
- **Source**: all_data.csv (200MB+ SafeBooru metadata)
- **Total Images**: 24,511 downloaded images
- **Classes**: 5 anime characters filtered by tags
  - Naruto (Naruto Shippuden)
  - Luffy (One Piece)
  - Gojo (Jujutsu Kaisen)
  - Goku (Dragon Ball)
  - Sukuna (Jujutsu Kaisen)
- **Split**: 80/20 train/val (17,280 / 4,321 images)
- **Labeling**: Haar cascade face detection for pseudo bounding boxes
- **Format**: YOLO format (class x_center y_center width height normalized)

### Data Pipeline Features
- ✅ Automatic download with retry logic (3 attempts)
- ✅ Image validation (removes GIFs and corrupted files)
- ✅ Tag-based filtering with substring matching
- ✅ Auto-labeling with face detection
- ✅ Stratified train/val split
- ✅ Albumentations augmentation (brightness, blur, noise, flip, scale)

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 16GB RAM (recommended for training)
- GPU with 6GB+ VRAM (optional but recommended)

### Installation
```powershell
# Clone repository
git clone https://github.com/username/Anime-YOLO-AI.git
cd Anime-YOLO-AI

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Full Pipeline (Automated)
```powershell
# 1. Download and prepare dataset (24,511 images)
python src\download_and_filter.py  # Downloads from all_data.csv
python src\validate_images.py      # Remove corrupted images
python src\autolabel_faces.py      # Generate pseudo labels
python src\split_dataset.py        # 80/20 train/val split

# 2. Run baseline benchmark
python src\benchmark.py            # EDA + logistic regression

# 3. Train YOLOv8 (50 epochs, ~2-3 hours on GPU)
python src\train.py

# 4. Evaluate model
python src\evaluate.py             # mAP, precision, recall, confusion matrix

# 5. Start API server
cd api
uvicorn main:app --host 0.0.0.0 --port 8000

# 6. Run load test (in separate terminal)
python src\load_test.py
```

## 🎓 Training

### Standard Training
```powershell
# Using train.py (recommended)
python src\train.py --epochs 50 --imgsz 640 --batch 16

# Direct YOLO CLI
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```

### MLflow Experiment Tracking
```powershell
# Train with MLflow logging
python src\train_mlflow.py

# View experiments
mlflow ui --port 5000
# Open http://localhost:5000
```

### Training Configuration
- **Model**: YOLOv8n (nano, 3.2M parameters)
- **Epochs**: 50
- **Image Size**: 640x640
- **Optimizer**: SGD (lr=0.01, momentum=0.9)
- **Augmentation**: Albumentations pipeline (built-in YOLO augmentation disabled)
- **Hardware**: CPU training ~5-7 hours, GPU (CUDA) ~2-3 hours

### Output Files
```
runs/detect/train/
├── weights/
│   ├── best.pt          # Best weights (highest mAP@0.5)
│   └── last.pt          # Last epoch weights
├── results.csv          # Per-epoch metrics
├── confusion_matrix.png # Class confusion visualization
├── PR_curve.png         # Precision-Recall curve
├── F1_curve.png         # F1 score curve
└── labels.jpg           # Label distribution visualization
```

## 🔍 Inference

### Webcam Real-time Detection
```powershell
python src\inference.py --weights runs/detect/train/weights/best.pt --source 0 --show
# Press 'q' to quit
```

### Image/Video Inference
```powershell
# Single image
python src\inference.py --weights runs/detect/train/weights/best.pt --source path\to\image.jpg --show

# Video file
python src\inference.py --weights runs/detect/train/weights/best.pt --source path\to\video.mp4 --show --save

# Directory of images
python src\inference.py --weights runs/detect/train/weights/best.pt --source path\to\images\ --save
```

### Batch Inference (Validation Set)
```powershell
# Process entire validation set and save annotated outputs
python src\batch_predict.py
# Outputs saved to runs/batch_preds/
```

## 📈 Evaluation & Metrics

### Comprehensive Evaluation
```powershell
python src\evaluate.py
```

**Metrics Computed**:
- mAP@0.5 (mean Average Precision at IoU=0.5)
- mAP@0.5:0.95 (COCO-style mAP across IoU thresholds)
- Precision (mean precision across all classes)
- Recall (mean recall across all classes)
- Per-class AP@0.5 for each character
- Confusion matrix visualization
- Precision-Recall curves

**Output**: `runs/evaluation/metrics.json`

### Training Metrics
During training, YOLOv8 automatically logs:
- Box loss (bounding box regression)
- Classification loss
- Distribution Focal Loss (DFL)
- Validation metrics every epoch

View metrics in `runs/detect/train/results.csv`

## 🎥 Object Tracking (Multi-Object)

Track characters with persistent IDs across video frames using ByteTrack:
```powershell
# Webcam tracking
python src\track.py --weights runs/detect/train/weights/best.pt --source 0 --show --save

# Video file tracking
python src\track.py --weights runs/detect/train/weights/best.pt --source path\to\video.mp4 --show --save
```

**Features**:
- Persistent object IDs across frames
- Re-identification after occlusion
- Multi-object association with IoU matching
- Outputs saved to `runs/track/`

## 🧪 Advanced Features

### Model Export (ONNX/TensorRT)
```powershell
# Export to ONNX (cross-platform inference)
yolo mode=export model=runs/detect/train/weights/best.pt format=onnx

# Export to TensorRT (NVIDIA GPUs, faster inference)
yolo mode=export model=runs/detect/train/weights/best.pt format=engine device=0
```

### Model Size Comparison
```powershell
# Compare YOLOv8n, YOLOv8s, YOLOv8m for speed/accuracy tradeoff
python src\sweep.py --models yolov8n.pt yolov8s.pt yolov8m.pt --epochs 5 --imgsz 640
```

### Baseline Benchmark (EDA)
```powershell
# Run exploratory data analysis + logistic regression baseline
python src\benchmark.py
```

**Outputs**:
- Dataset statistics (image count, label distribution)
- Tag frequency analysis
- Logistic regression baseline metrics (Accuracy, F1)
- Saved to `runs/benchmark/metrics.txt`

## 🔧 MLOps & CI/CD

### Experiment Tracking (MLflow)
```powershell
# Train with MLflow logging
python src\train_mlflow.py

# View experiments in UI
mlflow ui --port 5000
# Navigate to http://localhost:5000
```

**Tracked Artifacts**:
- Hyperparameters (model, epochs, imgsz, data_yaml)
- Training metrics (loss curves, mAP)
- Model weights (best.pt, last.pt)
- Confusion matrix and PR curves

### GitHub Actions CI/CD
Automated pipeline in `.github/workflows/ci.yml`:

**On Every Push/PR**:
- ✅ Python dependency installation with caching
- ✅ Dataset YAML validation
- ✅ Syntax check for all Python scripts
- ✅ Docker image build (smoke test)
- ✅ Unit tests with pytest and coverage reporting

**Nightly Schedule (2 AM UTC)**:
- 🌙 Download sample dataset
- 🌙 Train model for 5 epochs (CI validation)
- 🌙 Upload training artifacts

### Monitoring & Logging
- **Structured Logging**: All API requests logged with timestamp, latency, status
- **Metrics Endpoint**: `/metrics` for Prometheus scraping
- **Error Tracking**: Exception handling with detailed stack traces

### Data Versioning
- Dataset manifest tracked in `manifest_filtered.csv` (25,009 rows)
- Git LFS recommended for large model weights (not included in repo)

## 💡 Project Highlights (Level 3+ ML Engineering)

### Data Pipeline Excellence
- ✅ **Automated Data Collection**: SafeBooru API integration with retry logic
- ✅ **Data Quality**: Corruption detection, GIF filtering, image validation
- ✅ **Auto-Labeling**: Haar cascade pseudo-labeling for rapid dataset creation
- ✅ **Augmentation**: Albumentations pipeline (brightness, blur, noise, geometric transforms)
- ✅ **Versioning**: Manifest tracking with 25,009 labeled samples

### Model Development
- ✅ **Transfer Learning**: YOLOv8n pretrained on COCO (80 classes → 5 anime characters)
- ✅ **Baseline Comparison**: Logistic regression benchmark for sanity check
- ✅ **Evaluation Framework**: mAP@0.5, mAP@0.5:0.95, precision, recall, confusion matrix
- ✅ **Multi-Model Support**: Easy switching between YOLOv8n/s/m/l/x variants

### Production Deployment
- ✅ **REST API**: FastAPI with structured logging and latency tracking
- ✅ **Containerization**: Docker deployment with production-ready Dockerfile
- ✅ **Load Testing**: Async stress testing (100 req, 10 concurrent workers)
- ✅ **Monitoring**: Prometheus-compatible `/metrics` endpoint
- ✅ **Health Checks**: `/health` endpoint for orchestration

### MLOps Maturity
- ✅ **Experiment Tracking**: MLflow for hyperparameter logging and artifact versioning
- ✅ **CI/CD Pipeline**: GitHub Actions (tests, Docker build, nightly training)
- ✅ **Code Quality**: Syntax validation, pytest integration, coverage reporting
- ✅ **Reproducibility**: Fixed random seeds, versioned dependencies
- ✅ **Documentation**: Comprehensive README with examples and architecture details

### Advanced Capabilities
- ✅ **Object Tracking**: ByteTrack for persistent IDs in video streams
- ✅ **Model Export**: ONNX/TensorRT for edge deployment
- ✅ **Batch Inference**: Nightly prediction pipeline for validation monitoring
- ✅ **Multi-Source Inference**: Webcam, images, videos, directories

## 📚 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning Framework** | YOLOv8 (Ultralytics), PyTorch |
| **Computer Vision** | OpenCV 4.8+, Albumentations 1.4+ |
| **API Framework** | FastAPI 0.109+, Uvicorn 0.27+ |
| **Experiment Tracking** | MLflow 2.9+ |
| **Data Processing** | pandas 2.0+, NumPy 1.24+ |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Load Testing** | aiohttp 3.9+ (async) |
| **Visualization** | Matplotlib 3.8+, TQDM progress bars |
| **Python Version** | 3.11+ |

## 🎯 Performance Expectations

### Training Performance
- **Dataset**: 21,601 images (17,280 train + 4,321 val)
- **Training Time**: ~2-3 hours (GPU), ~5-7 hours (CPU)
- **Convergence**: Typically reaches mAP@0.5 > 0.85 by epoch 30-40
- **Hardware**: NVIDIA GPU with 6GB+ VRAM recommended

### Inference Performance
- **Latency**: <50ms per image (GPU), <200ms (CPU)
- **Throughput**: 20-30 FPS (GPU), 5-10 FPS (CPU)
- **API Response Time**: <100ms (excluding network latency)
- **Load Test Results**: 100 requests in ~10s (10 concurrent workers)

### Model Metrics (Expected)
- **mAP@0.5**: 0.80-0.90 (depends on label quality)
- **mAP@0.5:0.95**: 0.50-0.70
- **Precision**: 0.75-0.85
- **Recall**: 0.70-0.80
- **Model Size**: YOLOv8n = 6.2 MB (3.2M parameters)

## 🚀 Production Deployment

### FastAPI REST API

**Start Server**:
```powershell
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**API Endpoints**:
- `POST /predict` - Upload image for detection
  - Input: multipart/form-data with `file` field
  - Output: JSON with bboxes, classes, confidences
  - Logs: Request timestamp, latency, image dimensions
  
- `GET /health` - Health check endpoint
  - Returns: `{"status": "healthy"}`
  
- `GET /metrics` - Prometheus-compatible metrics
  - Returns: `service_up 1`

**Example cURL Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@test_image.jpg"
```

### Load Testing
```powershell
# Run async load test (100 requests, 10 concurrent)
python src\load_test.py
```

**Load Test Metrics**:
- Total requests: 100
- Concurrent workers: 10
- Latency statistics: min, max, avg, p95
- Success rate

### Docker Deployment
```powershell
# Build image
docker build -t anime-yolo-api:latest -f api/Dockerfile .

# Run container
docker run -d -p 8000:8000 anime-yolo-api:latest

# Test endpoint
curl http://localhost:8000/health
```

## 🚧 Future Enhancements

### Data Improvements
- [ ] Multi-source dataset fusion (Kaggle, Danbooru, Pixiv)
- [ ] Class balancing with oversampling/undersampling
- [ ] Hard negative mining for challenging examples
- [ ] Style transfer augmentation (sketch, watercolor, etc.)
- [ ] Active learning for label refinement

### Model Enhancements
- [ ] Multi-task learning (attributes: hair color, outfit, expressions)
- [ ] Fine-grained character disambiguation (Goku SSJ1 vs SSJ2)
- [ ] Cross-domain robustness (manga, fanart, screenshots)
- [ ] Quantization (INT8) for edge deployment
- [ ] Model distillation for mobile inference

### Production Features
- [ ] WebSocket streaming for real-time video
- [ ] Redis caching for frequently detected characters
- [ ] Batch processing queue (RabbitMQ/Celery)
- [ ] A/B testing framework for model versions
- [ ] Explainability (Grad-CAM heatmaps)

### MLOps Upgrades
- [ ] DVC for large dataset versioning
- [ ] Kubeflow/MLflow for model registry
- [ ] Auto-retraining on data drift detection
- [ ] Feature store for character embeddings
- [ ] Shadow deployment for canary releases

## 🤝 Contributing

Contributions welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics) for the detection framework
- **SafeBooru**: Dataset source for anime character images
- **FastAPI**: High-performance API framework
- **MLflow**: Experiment tracking and model registry

## 📧 Contact

For questions or collaboration:
- GitHub: [@username](https://github.com/username)
- Email: your.email@example.com

---

**Resume Line**: *"Developed production-grade anime character detection system using YOLOv8 with full MLOps pipeline (MLflow tracking, FastAPI deployment, CI/CD, Docker containerization), achieving real-time inference on 24K+ image dataset with comprehensive evaluation framework."*
