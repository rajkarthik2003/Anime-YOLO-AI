#!/usr/bin/env python3
"""
Data Cleaning Script
Removes images without labels and creates clean dataset
Handles images from non-target characters
"""

from pathlib import Path
import shutil
import json

print("=" * 70)
print("🧹 DATA CLEANING SCRIPT")
print("=" * 70)

def clean_dataset():
    """Remove unlabeled images and organize dataset"""
    
    # Datasets to clean
    for split in ['train', 'val', 'test']:
        img_dir = Path(f'data/raw/images/{split}')
        label_dir = Path(f'data/raw/labels/{split}')
        
        if not img_dir.exists() or not label_dir.exists():
            continue
            
        print(f"\n📁 Cleaning {split} set...")
        
        # Find all images
        images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        print(f"   Total images: {len(images)}")
        
        # Track images without labels
        unlabeled = []
        labeled = []
        empty_labels = []
        
        for img_path in images:
            label_path = label_dir / (img_path.stem + '.txt')
            
            if not label_path.exists():
                unlabeled.append(img_path)
            else:
                # Check if label file is empty
                with open(label_path) as f:
                    content = f.read().strip()
                if not content:
                    empty_labels.append(img_path)
                else:
                    labeled.append(img_path)
        
        print(f"   ✅ Labeled (with content): {len(labeled)}")
        print(f"   ⚠️  Unlabeled (no .txt file): {len(unlabeled)}")
        print(f"   ⚠️  Empty labels (no objects): {len(empty_labels)}")
        
        # Option: Remove unlabeled images (OPTIONAL - do this only if needed)
        # Uncomment below to actually delete unlabeled images
        """
        if unlabeled or empty_labels:
            print(f"   🗑️  Removing {len(unlabeled) + len(empty_labels)} unlabeled/empty images...")
            for img in unlabeled + empty_labels:
                img.unlink(missing_ok=True)
                (label_dir / (img.stem + '.txt')).unlink(missing_ok=True)
        """
    
    return {
        "message": "Data analysis complete. Unlabeled images identified but not deleted.",
        "recommendation": "Consider removing unlabeled/empty images to focus training on annotated data"
    }

def analyze_labels():
    """Analyze label distribution"""
    print("\n" + "=" * 70)
    print("📊 LABEL DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    class_names = {0: 'naruto', 1: 'luffy', 2: 'gojo', 3: 'goku', 4: 'sukuna'}
    class_counts = {cls: 0 for cls in class_names.values()}
    total_objects = 0
    
    for split in ['train', 'val']:
        label_dir = Path(f'data/raw/labels/{split}')
        if not label_dir.exists():
            continue
            
        for label_file in label_dir.glob('*.txt'):
            with open(label_file) as f:
                for line in f:
                    if line.strip():
                        cls_id = int(line.split()[0])
                        if cls_id in class_names:
                            class_counts[class_names[cls_id]] += 1
                            total_objects += 1
    
    print(f"\nTotal objects across dataset: {total_objects}")
    print("\nClass distribution:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / total_objects if total_objects > 0 else 0
        bar = '█' * int(pct / 5)
        print(f"  {cls_name:10s}: {count:6d} ({pct:5.1f}%) {bar}")
    
    return class_counts

if __name__ == '__main__':
    # Run analysis
    result = clean_dataset()
    class_dist = analyze_labels()
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nRecommendations:")
    print("1. Remove unlabeled images to focus training on annotated data")
    print("2. Some characters may be underrepresented - consider data augmentation")
    print("3. Use the improved training script: python src/train_improved.py")
    print("=" * 70)
