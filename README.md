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
- Export labels in YOLO format (class x_center y_center width height normalized).
- Use `src/data_prep.py` to augment and split 80/20 into `data/images/{train,val}` and `data/labels/{train,val}`.
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
 - Benchmark FPS:
```powershell
python src\benchmark.py --weights runs/detect/train/weights/best.pt --source 0 --measure 300
```

## Dataset Preparation Workflow
1. Collect and label images (Roboflow or LabelImg), export in YOLO format.
2. Place raw images and labels in folders, e.g. `raw/images` and `raw/labels`.
3. Run augmentation and split:
```powershell
python src\data_prep.py --in-images raw\images --in-labels raw\labels --out-images prepared\images --out-labels prepared\labels --aug-per-image 1 --train-ratio 0.8 --update-yaml --nc 5 --names Naruto Goku Luffy Sukuna Gojo
```
4. Verify `data/images/{train,val}` and `data/labels/{train,val}` were created.
5. Start training as above.

### Using a Public Roboflow Dataset
1. Find a dataset on Roboflow Universe: search "Anime Object Detection" or "Anime Face Detection YOLO".
2. Export as YOLOv8 (PyTorch) format and get a direct download URL.
3. Download and place it automatically:
```powershell
python src\download_dataset.py --url "<direct-zip-download-url>"
```
4. Check `data/images/train`, `data/images/val`, `data/labels/train`, `data/labels/val`.
5. Update `dataset.yaml` classes (`nc` and `names`) to match your dataset.

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
