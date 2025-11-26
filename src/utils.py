import os
import shutil
from pathlib import Path
from typing import Tuple
import random


def make_dirs(base: str) -> None:
    Path(base, 'data', 'images', 'train').mkdir(parents=True, exist_ok=True)
    Path(base, 'data', 'images', 'val').mkdir(parents=True, exist_ok=True)
    Path(base, 'data', 'labels', 'train').mkdir(parents=True, exist_ok=True)
    Path(base, 'data', 'labels', 'val').mkdir(parents=True, exist_ok=True)


def split_dataset(images_dir: str, labels_dir: str, train_ratio: float = 0.8) -> Tuple[int, int]:
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    def copy_subset(subset, subset_name):
        for img in subset:
            src_img = Path(images_dir, img)
            src_label = Path(labels_dir, Path(img).with_suffix('.txt').name)
            if not src_label.exists():
                # Skip images without labels
                continue
            dst_img = Path('data/images', subset_name, img)
            dst_label = Path('data/labels', subset_name, src_label.name)
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_label, dst_label)

    copy_subset(train_images, 'train')
    copy_subset(val_images, 'val')
    return len(train_images), len(val_images)
