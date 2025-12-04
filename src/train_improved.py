#!/usr/bin/env python3
"""
Improved Training Script - Better Accuracy
Uses YOLOv8 Medium (7.1M params) instead of Nano
Trains for 100 epochs with data augmentation
"""

from ultralytics import YOLO
import json
from pathlib import Path

print("=" * 70)
print("🚀 ANIME-YOLO-AI - IMPROVED TRAINING (YOLOv8 Medium)")
print("=" * 70)

# Load pre-trained medium model (larger, more accurate)
print("\n[1/3] Loading YOLOv8 Medium model...")
model = YOLO('yolov8m.pt')  # Medium model: 25.9M params, 78.9M FLOPs
print("✅ YOLOv8m loaded")
print("   Parameters: 25,900,000 (vs 3M for nano)")
print("   FLOPs: 78.9M (vs 8.1M for nano)")

# Train with better hyperparameters
print("\n[2/3] Training with improved settings...")
print("   - Epochs: 100 (vs 50)")
print("   - Image size: 640x640")
print("   - Batch size: 16")
print("   - Learning rate: Auto (0.01 start)")
print("   - Augmentation: Heavy (mosaic, mixup, flip)")
print("   - Device: CPU")

results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,  # Early stopping if no improvement for 20 epochs
    device=0,  # Use GPU if available, else CPU
    save=True,
    augment=True,
    mosaic=1.0,
    flipud=0.5,
    fliplr=0.5,
    degrees=10,
    translate=0.1,
    scale=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    optimizer='SGD',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    project='runs',
    name='detect_improved',
    pretrained=True,
    verbose=True,
    seed=42,
)

print("\n[3/3] Training complete!")
print(f"✅ Best weights saved to: {results.save_dir}/weights/best.pt")

# Evaluate
print("\n" + "=" * 70)
print("📊 EVALUATING MODEL")
print("=" * 70)

best_model = YOLO(f'{results.save_dir}/weights/best.pt')
val_results = best_model.val()

print(f"\n✅ Validation Results:")
print(f"   mAP@0.5:      {val_results.box.map50:.3f}")
print(f"   Precision:    {val_results.box.mp:.3f}")
print(f"   Recall:       {val_results.box.mr:.3f}")
print(f"   mAP@0.5:0.95: {val_results.box.map:.3f}")

# Save metrics
metrics = {
    "model": "YOLOv8m (Medium)",
    "epochs": 100,
    "parameters": "25.9M",
    "box_metrics": {
        "mAP50": float(val_results.box.map50),
        "mAP50-95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr)
    }
}

metrics_dir = Path('runs/evaluation_improved')
metrics_dir.mkdir(parents=True, exist_ok=True)
with open(metrics_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Metrics saved to: runs/evaluation_improved/metrics.json")

print("\n" + "=" * 70)
print("🎉 IMPROVED TRAINING COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("1. Compare accuracy: runs/detect_improved/weights/best.pt vs runs/detect/train2/weights/best.pt")
print("2. Update API to use new model if better")
print("3. Test with your Naruto screenshot again")
print("=" * 70)
