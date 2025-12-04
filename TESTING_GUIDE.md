# 🧪 Testing Guide - Anime-YOLO-AI

## Quick Tests You Can Run

### 1. **Automated Full Test** ✅ (Recommended)
```powershell
python test_project.py
```
**What it tests:** Model loading, evaluation metrics, inference, API imports, project structure

---

### 2. **Interactive Demo** 🎯
```powershell
python demo_test.py
```
**What it shows:** 
- Model performance metrics
- Live inference on validation images
- Detection examples with confidence scores
- API readiness check

---

### 3. **Test on Specific Image** 🖼️
```powershell
# Find an image in validation set
$img = (Get-ChildItem "data\raw\images\val\*.jpg" | Select-Object -First 1).FullName

# Run inference
python src/inference.py --source $img --conf 0.25
```

**Example with actual image:**
```powershell
python src/inference.py --source "data/raw/images/val/1011503_sample_dfaa5deed63830d1435bf66dedd867448abdf594.jpg" --conf 0.25
```

---

### 4. **Test with Webcam** 📷 (Real-time Detection)
```powershell
python src/inference.py --source 0 --show
```
**Controls:**
- Press `q` to quit
- Model will detect anime characters in real-time

---

### 5. **Test API Endpoint** 🌐

**Start the server:**
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Test in browser:**
- Health check: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics
- API docs: http://localhost:8000/docs

**Test with PowerShell:**
```powershell
# Test health endpoint
Invoke-WebRequest -Uri http://localhost:8000/health

# Test prediction (requires multipart form data)
# Use the interactive docs at http://localhost:8000/docs
```

---

### 6. **Batch Prediction Test** 📦
```powershell
python src/batch_predict.py
```
**Output:** Annotated images saved in `runs/batch_preds/`

---

### 7. **Evaluation Test** 📊
```powershell
python src/evaluate.py
```
**Output:** Metrics saved to `runs/evaluation/metrics.json`

---

## Test Results You Should See

### ✅ Successful Model Inference:
```
image 1/1 ...: 640x640 3 narutos, 154.6ms
Speed: 6.4ms preprocess, 154.6ms inference, 3.0ms postprocess
```

### ✅ Model Performance:
- **mAP@0.5:** 0.223 (22.3%)
- **Precision:** 0.373 (37.3%)
- **Recall:** 0.225 (22.5%)

### ✅ All Automated Tests:
```
[TEST 1] Model weights: PASSED
[TEST 2] Model loading: PASSED
[TEST 3] Evaluation metrics: PASSED
[TEST 4] Inference pipeline: PASSED
[TEST 5] Dataset config: PASSED
[TEST 6] Project structure: PASSED
[TEST 7] API imports: PASSED
```

---

## Common Commands Summary

| Task | Command |
|------|---------|
| Full automated test | `python test_project.py` |
| Interactive demo | `python demo_test.py` |
| Webcam detection | `python src/inference.py --source 0 --show` |
| Image detection | `python src/inference.py --source path/to/image.jpg` |
| API server | `python -m uvicorn api.main:app --port 8000` |
| Batch predictions | `python src/batch_predict.py` |
| Model evaluation | `python src/evaluate.py` |

---

## What to Expect

### Model Behavior:
- The model detects anime characters (Naruto, Luffy, Gojo, Goku, Sukuna)
- With `--conf 0.25` (default), it's conservative to avoid false positives
- Lower confidence (`--conf 0.1`) will detect more but with more false positives
- Some validation images may have no detections (normal for conservative model)

### Performance:
- **CPU Inference:** ~70-150ms per image
- **Throughput:** ~6-7 images/second
- **Best for:** Images with clear anime character faces

---

## Troubleshooting

### "FileNotFoundError"
➡️ Make sure you use a real image path, not placeholder text like "your_image.jpg"

### "No detections"
➡️ Try lower confidence: `--conf 0.1`
➡️ Model is trained on specific anime characters, may not detect all anime

### API won't start
➡️ Use: `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`
➡️ Check if port 8000 is already in use

---

## 🎯 Ready to Showcase!

Your project includes:
- ✅ Trained YOLOv8 model
- ✅ Comprehensive testing suite
- ✅ Real-time inference capability
- ✅ REST API endpoint
- ✅ Docker deployment setup
- ✅ CI/CD pipeline
- ✅ Full documentation

**Perfect for demonstrating ML engineering skills in job applications!** 🚀
