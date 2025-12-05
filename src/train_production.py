#!/usr/bin/env python3
"""
PRODUCTION-READY MULTI-CLASS TRAINING
Starts immediately with current dataset or defaults
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import json

print("=" * 80)
print("🚀 PRODUCTION-READY MULTI-CLASS ANIME CHARACTER DETECTION")
print("=" * 80)

# Check dataset exists
dataset_path = Path('dataset.yaml')
if not dataset_path.exists():
    print("⚠️  WARNING: dataset.yaml not found!")
    print("   Will create default multi-class config...")
    
    # Create default dataset
    default_yaml = """path: .
train: data/images/train
val: data/images/val

nc: 10
names:
  0: naruto
  1: goku
  2: luffy
  3: gojo
  4: sasuke
  5: sukuna
  6: kakashi
  7: tanjiro
  8: zoro
  9: megumi
"""
    with open('dataset.yaml', 'w') as f:
        f.write(default_yaml)
    print("   ✓ Created dataset.yaml with 10 default classes")

print(f"\n{'='*80}")
print("TRAINING CONFIGURATION")
print(f"{'='*80}")

config = {
    'model': 'yolov8l.pt',          # Large model (43.7M params)
    'dataset': 'dataset.yaml',
    'epochs': 120,
    'batch_size': 32,
    'img_size': 640,
    'device': 0,                    # GPU 0
    'augment': True,
    'patience': 25,                 # Early stopping
    'save': True,
    'plots': True,
}

print(f"\n📊 Model Configuration:")
print(f"   Model:       {config['model']}")
print(f"   Dataset:     {config['dataset']}")
print(f"   Epochs:      {config['epochs']}")
print(f"   Batch size:  {config['batch_size']}")
print(f"   Image size:  {config['img_size']}x{config['img_size']}")
print(f"\n⚡ Advanced Settings:")
print(f"   Augmentation: Enabled (mosaic, mixup, flip, rotate)")
print(f"   Early stop:   Yes (patience: {config['patience']} epochs)")
print(f"   Device:       GPU 0")

print(f"\n{'='*80}")
print("STARTING TRAINING")
print(f"{'='*80}\n")

try:
    # Load model
    print(f"[1/3] Loading YOLOv8 Large...")
    model = YOLO(config['model'])
    
    # Train with advanced hyperparameters
    print(f"[2/3] Starting training ({config['epochs']} epochs)...")
    results = model.train(
        data=config['dataset'],
        epochs=config['epochs'],
        imgsz=config['img_size'],
        batch=config['batch_size'],
        device=config['device'],
        augment=config['augment'],
        
        # Advanced augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=15,
        translate=0.15,
        scale=0.7,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Optimization
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        
        # Loss weights
        box=7.5,
        cls=3.0,
        dfl=1.5,
        
        # Validation & early stop
        val=True,
        patience=config['patience'],
        plots=config['plots'],
        
        # Output
        project='runs',
        name='detect_multiclass_production',
        exist_ok=False,
        
        # Reproducibility
        seed=42,
        deterministic=True,
        verbose=True,
    )
    
    print(f"\n[3/3] Training complete! ✓")
    
    # Evaluation
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}\n")
    
    best_model = YOLO(f'{results.save_dir}/weights/best.pt')
    val_results = best_model.val()
    
    # Display key metrics
    metrics = {
        'mAP@0.5': float(val_results.box.map50),
        'mAP@0.5:0.95': float(val_results.box.map),
        'Precision': float(val_results.box.mp),
        'Recall': float(val_results.box.mr),
    }
    
    print("📊 FINAL METRICS:")
    print(f"   mAP@0.5:      {metrics['mAP@0.5']:.3f}")
    print(f"   mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.3f}")
    print(f"   Precision:    {metrics['Precision']:.3f}")
    print(f"   Recall:       {metrics['Recall']:.3f}")
    
    print(f"\n⚡ INFERENCE SPEED:")
    print(f"   Preprocess:   {val_results.speed[0]:.1f}ms")
    print(f"   Inference:    {val_results.speed[1]:.1f}ms")
    print(f"   Postprocess:  {val_results.speed[2]:.1f}ms")
    
    # Save metrics
    metrics_file = Path(results.save_dir) / 'metrics' / 'results.json'
    metrics_file.parent.mkdir(exist_ok=True)
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ TRAINING SUCCESSFUL")
    print(f"{'='*80}")
    print(f"\n📁 Output files:")
    print(f"   Weights:      {results.save_dir}/weights/best.pt")
    print(f"   Metrics:      {metrics_file}")
    print(f"   Plots:        {results.save_dir}/")
    
    print(f"\n🧪 Next steps:")
    print(f"   # Test on image")
    print(f"   python src/inference.py --image test.jpg \\")
    print(f"       --model {results.save_dir}/weights/best.pt")
    print(f"\n   # Test with API")
    print(f"   python api/main.py")
    print(f"\n   # Upload screenshot to http://localhost:8000/docs")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
