import argparse
import os
from pathlib import Path
import zipfile
import shutil
import sys
import urllib.request


def download_zip(url: str, out_path: Path):
    print(f'Downloading from {url} to {out_path}')
    with urllib.request.urlopen(url) as resp, open(out_path, 'wb') as f:
        shutil.copyfileobj(resp, f)


def extract_and_place(zip_path: Path, project_root: Path):
    tmp_dir = project_root / 'tmp_extract'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmp_dir)
    # Try to find YOLOv8 layout from Roboflow export (train/valid/test with images/labels)
    # Normalize to our layout: data/images/train, data/images/val, data/labels/train, data/labels/val
    def copy_tree(src_img_dir, src_lbl_dir, subset):
        dst_img = project_root / 'data' / 'images' / subset
        dst_lbl = project_root / 'data' / 'labels' / subset
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        if src_img_dir and os.path.isdir(src_img_dir):
            for root, _, files in os.walk(src_img_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        shutil.copy2(Path(root) / f, dst_img / f)
        if src_lbl_dir and os.path.isdir(src_lbl_dir):
            for root, _, files in os.walk(src_lbl_dir):
                for f in files:
                    if f.lower().endswith('.txt'):
                        shutil.copy2(Path(root) / f, dst_lbl / f)

    # Common Roboflow export patterns
    candidates = [
        # Roboflow YOLOv8 default
        {
            'train_img': tmp_dir / 'train' / 'images',
            'train_lbl': tmp_dir / 'train' / 'labels',
            'val_img': tmp_dir / 'valid' / 'images',
            'val_lbl': tmp_dir / 'valid' / 'labels',
        },
        # Alternate naming
        {
            'train_img': tmp_dir / 'train',
            'train_lbl': tmp_dir / 'train' / 'labels',
            'val_img': tmp_dir / 'valid',
            'val_lbl': tmp_dir / 'valid' / 'labels',
        },
    ]
    matched = None
    for c in candidates:
        if c['train_img'].exists():
            matched = c
            break
    if matched is None:
        print('Could not locate YOLOv8 folders in zip. Please inspect tmp_extract.')
    else:
        copy_tree(matched['train_img'], matched['train_lbl'], 'train')
        copy_tree(matched['val_img'], matched['val_lbl'], 'val')

    # Clean up
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description='Download and unpack YOLOv8 dataset zip (Roboflow export)')
    ap.add_argument('--url', type=str, help='Direct download URL to YOLOv8 dataset zip')
    ap.add_argument('--zip', type=str, help='Local zip path if already downloaded')
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    zip_path = None
    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            print(f'Zip not found: {zip_path}')
            sys.exit(1)
    elif args.url:
        zip_path = project_root / 'dataset.zip'
        download_zip(args.url, zip_path)
    else:
        print('Provide --url or --zip')
        sys.exit(1)

    extract_and_place(zip_path, project_root)
    print('Dataset placed under data/images and data/labels.')


if __name__ == '__main__':
    main()