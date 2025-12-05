#!/usr/bin/env python3
"""
ADVANCED TRAINING SCRIPT FOR HIGH ACCURACY
Multi-class anime character detection with best practices
"""

from ultralytics import YOLO
import json
from pathlib import Path
import numpy as np

print("=" * 70)
print("🚀 ADVANCED TRAINING - MAXIMUM ACCURACY")
print("=" * 70)

def train_multiclass(
    dataset_yaml='dataset.yaml',
    model_size='large',  # nano, small, medium, large, xlarge
    epochs=150,
    batch_size=32,
    img_size=640,
    num_classes=10,
):
    """
    Train with advanced hyperparameters for best accuracy
    """
    
    # Model selection
    models = {
        'nano': ('yolov8n.pt', 3.0),
        'small': ('yolov8s.pt', 11.2),
        'medium': ('yolov8m.pt', 25.9),
        'large': ('yolov8l.pt', 43.7),
        'xlarge': ('yolov8x.pt', 68.2),
    }
    
    model_path, params_m = models[model_size]
    
    print(f"\n📊 CONFIGURATION:")
    print(f"   Model: YOLOv8 {model_size.upper()}")
    print(f"   Parameters: {params_m}M")
    print(f"   Classes: {num_classes}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {img_size}x{img_size}")
    
    # Load model
    print(f"\n[1/3] Loading model...")
    model = YOLO(model_path)
    
    # Advanced hyperparameters for better accuracy
    print(f"\n[2/3] Starting training with advanced augmentation...")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=25,
        device=0,
        save=True,
        augment=True,
        
        # Strong augmentation for multi-class
        mosaic=1.0,              # Mosaic augmentation
        mixup=0.1,               # Mixup augmentation
        copy_paste=0.1,          # Copy-paste augmentation
        
        # Geometric transforms
        degrees=15,              # Rotation
        translate=0.15,          # Translation
        scale=0.7,               # Scaling
        flipud=0.5,              # Vertical flip
        fliplr=0.5,              # Horizontal flip
        perspective=0.0001,      # Perspective transform
        
        # Color transforms
        hsv_h=0.015,             # HSV hue
        hsv_s=0.7,               # HSV saturation
        hsv_v=0.4,               # HSV value
        
        # Training optimization
        optimizer='SGD',         # SGD often better than Adam for detection
        lr0=0.01,                # Initial learning rate
        lrf=0.01,                # Final learning rate ratio
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Loss weights (focus on box and class loss)
        box=7.5,
        cls=3.0,
        dfl=1.5,
        
        # Validation
        val=True,
        plots=True,
        
        # Project settings
        project='runs',
        name='detect_multiclass_v1',
        exist_ok=False,
        
        # Reproducibility
        seed=42,
        deterministic=True,
        verbose=True,
    )
    
    print(f"\n[3/3] Training complete!")
    
    # Evaluation
    print(f"\n{'='*70}")
    print(f"📊 VALIDATION RESULTS")
    print(f"{'='*70}")
    
    best_model = YOLO(f'{results.save_dir}/weights/best.pt')
    val_results = best_model.val()
    
    # Display results
    results_dict = {
        "model": f"YOLOv8 {model_size.upper()}",
        "parameters_M": params_m,
        "classes": num_classes,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": img_size,
        "box_metrics": {
            "mAP50": float(val_results.box.map50),
            "mAP50-95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
        },
        "speed": {
            "preprocess_ms": float(val_results.speed[0]),
            "inference_ms": float(val_results.speed[1]),
            "postprocess_ms": float(val_results.speed[2]),
        }
    }
    
    print(f"\n✅ KEY METRICS:")
    print(f"   mAP@0.5:      {results_dict['box_metrics']['mAP50']:.3f}")
    print(f"   mAP@0.5:0.95: {results_dict['box_metrics']['mAP50-95']:.3f}")
    print(f"   Precision:    {results_dict['box_metrics']['precision']:.3f}")
    print(f"   Recall:       {results_dict['box_metrics']['recall']:.3f}")
    
    print(f"\n⚡ INFERENCE SPEED:")
    print(f"   Preprocess:   {results_dict['speed']['preprocess_ms']:.1f}ms")
    print(f"   Inference:    {results_dict['speed']['inference_ms']:.1f}ms")
    print(f"   Postprocess:  {results_dict['speed']['postprocess_ms']:.1f}ms")
    print(f"   Total:        {sum(results_dict['speed'].values()):.1f}ms")
    
    # Save detailed results
    metrics_dir = Path(results.save_dir) / 'metrics'
    metrics_dir.mkdir(exist_ok=True)
    
    with open(metrics_dir / 'results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n📁 Results saved to: {results.save_dir}")
    print(f"   Weights: {results.save_dir}/weights/best.pt")
    print(f"   Metrics: {metrics_dir}/results.json")
    
    return results_dict

if __name__ == '__main__':
    print("\n" + "="*70)
    print("SELECT TRAINING CONFIGURATION")
    print("="*70)
    
    configs = {
        '1': {
            'name': '10 Classes - High Accuracy (Recommended)',
            'model_size': 'large',
            'epochs': 120,
            'batch_size': 32,
            'num_classes': 10,
        },
        '2': {
            'name': '15 Classes - Balanced',
            'model_size': 'xlarge',
            'epochs': 150,
            'batch_size': 24,
            'num_classes': 15,
        },
        '3': {
            'name': '20 Classes - Maximum Coverage',
            'model_size': 'xlarge',
            'epochs': 200,
            'batch_size': 16,
            'num_classes': 20,
        },
    }
    
    for key, config in configs.items():
        print(f"\n{key}. {config['name']}")
        print(f"   Model: YOLOv8 {config['model_size'].upper()}")
        print(f"   Classes: {config['num_classes']}")
        print(f"   Time: {config['epochs'] * 0.5 / 60:.1f} hours (CPU)")
    
    print("\n" + "="*70)
    print("TRAINING RESULTS COMPARISON")
    print("="*70)
    
    comparison = {
        'Config': ['10 Classes', '15 Classes', '20 Classes'],
        'Model': ['Large', 'XLarge', 'XLarge'],
        'Expected mAP@0.5': ['40-45%', '30-35%', '25-30%'],
        'Inference': ['180ms', '200ms', '250ms'],
        'Best For': ['Production', 'Balanced', 'Coverage'],
    }
    
    print("\n" + " | ".join(f"{k:15s}" for k in comparison.keys()))
    print("-" * 100)
    for i in range(len(comparison['Config'])):
        print(" | ".join(f"{comparison[k][i]:15s}" for k in comparison.keys()))
    
    print("\n" + "="*70)
    print("🎯 RECOMMENDATION: Use Configuration #1 (10 Classes)")
    print("   - Best accuracy (40-45% mAP)")
    print("   - Reasonable training time (60 hours on CPU)")
    print("   - Production-ready speed (180ms)")
    print("="*70)
    
    # Example: Train 10-class model
    print("\n✨ To train 10-class model, run:")
    print("""
    results = train_multiclass(
        dataset_yaml='dataset.yaml',
        model_size='large',
        epochs=120,
        batch_size=32,
        num_classes=10,
    )
    """)
