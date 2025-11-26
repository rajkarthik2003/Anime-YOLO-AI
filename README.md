# Anime YOLO AI

Real-time anime character detection using YOLOv8, PyTorch, and OpenCV.

## Project Overview
- Detects and classifies anime characters in images, videos, and webcam.
- Transfer learning from COCO-pretrained YOLOv8.
- Includes preprocessing, augmentation, training, and evaluation (mAP, precision, recall, FPS).

## Folder Structure
```
Anime-YOLO-AI/
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── models/
├── runs/
├── src/
│   ├── train.py
│   ├── inference.py
│   └── utils.py
├── requirements.txt
├── README.md
└── dataset.yaml
```

## Dataset
- Recommended: Kaggle "Anime Face Detection" datasets (5k–10k images).
- Label via Roboflow (web) or LabelImg (local).
- Split 80/20 into `data/images/{train,val}` and `data/labels/{train,val}` with YOLO-format labels.
- Update `dataset.yaml` with character names and `nc`.

## Setup
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify YOLO installation:
```powershell
yolo version
```

## Training
Quick train via CLI:
```powershell
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```
Or using `src/train.py`:
```powershell
python src\train.py --model yolov8n.pt --data dataset.yaml --epochs 50 --imgsz 640 --batch 16 --workers 4
```

## Inference
Webcam (press `q` to quit):
```powershell
python src\inference.py --weights runs/detect/train/weights/best.pt --source 0 --show
```
Image or video path:
```powershell
python src\inference.py --weights runs/detect/train/weights/best.pt --source path\to\file.jpg --show
```

## Evaluation
- mAP@0.5, Precision, Recall auto-reported by YOLO during training.
- Inspect `runs/detect/train` for `results.csv`, `confusion_matrix.png`, and PR curves.
- Report FPS from webcam inference by measuring loop time.

## Deployment
- Package inference as a Python script using OpenCV.
- Optionally export to `onnx` for deployment on other runtimes:
```powershell
yolo mode=export model=runs/detect/train/weights/best.pt format=onnx
```

## Future Improvements
- More classes or fine-grained character sets.
- Data balancing and hard-negative mining.
- Mixed precision training and quantization-aware export.

## Resume Line
"Developed real-time anime character detection system using YOLOv8, achieving high accuracy and real-time performance through transfer learning and robust data augmentation."
