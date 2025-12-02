"""
Model Evaluation Script - Comprehensive metrics for trained YOLOv8 model
Computes precision, recall, mAP@0.5, mAP@0.5:0.95, confusion matrix
"""

import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json
from pathlib import Path

def evaluate_model(weights_path='runs/detect/train/weights/best.pt', 
                   data_yaml='dataset.yaml',
                   output_dir='runs/evaluation'):
    """
    Run comprehensive evaluation on validation set
    
    Args:
        weights_path: Path to trained model weights
        data_yaml: Path to dataset configuration
        output_dir: Directory to save evaluation outputs
    
    Returns:
        dict: Evaluation metrics
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)
    
    print("Running validation with metrics computation...")
    # Ultralytics YOLO.val() computes all metrics automatically
    metrics = model.val(data=data_yaml, 
                        save_json=True,  # Save COCO JSON for detailed analysis
                        save_hybrid=True,  # Save hybrid labels
                        conf=0.001,  # Low confidence for full precision-recall curve
                        iou=0.6,
                        plots=True)
    
    # Extract key metrics
    results = {
        'box_metrics': {
            'mAP50': float(metrics.box.map50),  # mAP @ IoU=0.5
            'mAP50-95': float(metrics.box.map),  # mAP @ IoU=0.5:0.95
            'precision': float(metrics.box.mp),  # mean precision
            'recall': float(metrics.box.mr),  # mean recall
        },
        'per_class_metrics': {}
    }
    
    # Per-class metrics
    class_names = ['naruto', 'luffy', 'gojo', 'goku', 'sukuna']
    for i, cls_name in enumerate(class_names):
        if i < len(metrics.box.ap):
            results['per_class_metrics'][cls_name] = {
                'AP50': float(metrics.box.ap[i]),
                'AP50-95': float(metrics.box.ap_class_index.get(i, 0.0))
            }
    
    # Save metrics to JSON
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"mAP@0.5:      {results['box_metrics']['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {results['box_metrics']['mAP50-95']:.4f}")
    print(f"Precision:    {results['box_metrics']['precision']:.4f}")
    print(f"Recall:       {results['box_metrics']['recall']:.4f}")
    print(f"\nPer-class AP@0.5:")
    for cls_name, metrics_dict in results['per_class_metrics'].items():
        print(f"  {cls_name:10s}: {metrics_dict['AP50']:.4f}")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Metrics JSON: {metrics_file}")
    
    return results

if __name__ == '__main__':
    # Check if weights exist
    weights_path = 'runs/detect/train/weights/best.pt'
    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found at {weights_path}")
        print("Please train the model first using train.py")
        exit(1)
    
    evaluate_model()
