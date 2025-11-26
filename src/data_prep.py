import argparse
import os
from pathlib import Path
import shutil
import random
import yaml
import cv2
import albumentations as A

# YOLO-format augmentation: transforms image and adjusts bbox in-place
# BBoxes are in Pascal VOC format for Albumentations (x_min, y_min, x_max, y_max), class label separate.
# We will convert from YOLO txt (class x_center y_center width height [normalized]) → VOC → apply → back to YOLO.

def yolo_to_voc(xc, yc, w, h, img_w, img_h):
    x = (xc - w/2) * img_w
    y = (yc - h/2) * img_h
    x2 = (xc + w/2) * img_w
    y2 = (yc + h/2) * img_h
    return [x, y, x2, y2]


def voc_to_yolo(x, y, x2, y2, img_w, img_h):
    w = (x2 - x) / img_w
    h = (y2 - y) / img_h
    xc = (x + x2) / (2 * img_w)
    yc = (y + y2) / (2 * img_h)
    return [xc, yc, w, h]


def parse_label_file(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            boxes.append((cls, xc, yc, w, h))
    return boxes


def write_label_file(label_path, boxes):
    with open(label_path, 'w') as f:
        for cls, xc, yc, w, h in boxes:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def build_augmenter():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), rotate=(-5, 5), p=0.5),
        A.ColorJitter(p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def augment_dataset(images_dir, labels_dir, out_images_dir, out_labels_dir, aug_per_image=1):
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)
    Path(out_labels_dir).mkdir(parents=True, exist_ok=True)
    aug = build_augmenter()

    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in images:
        img_path = Path(images_dir, img_name)
        label_path = Path(labels_dir, Path(img_name).with_suffix('.txt').name)
        if not label_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        boxes_raw = parse_label_file(label_path)
        bboxes = []
        labels = []
        for cls, xc, yc, bw, bh in boxes_raw:
            x, y, x2, y2 = yolo_to_voc(xc, yc, bw, bh, w, h)
            bboxes.append([x, y, x2, y2])
            labels.append(cls)

        # Save original
        shutil.copy2(img_path, Path(out_images_dir, img_name))
        shutil.copy2(label_path, Path(out_labels_dir, label_path.name))

        # Augmentations
        for k in range(aug_per_image):
            try:
                transformed = aug(image=image, bboxes=bboxes, labels=labels)
                img_t = transformed['image']
                bboxes_t = transformed['bboxes']
                labels_t = transformed['labels']

                # Convert back to YOLO and write
                yolo_boxes = []
                for (x, y, x2, y2), cls in zip(bboxes_t, labels_t):
                    # Clip to image
                    x = max(0, min(x, img_t.shape[1]-1))
                    y = max(0, min(y, img_t.shape[0]-1))
                    x2 = max(0, min(x2, img_t.shape[1]-1))
                    y2 = max(0, min(y2, img_t.shape[0]-1))
                    xc, yc, bw, bh = voc_to_yolo(x, y, x2, y2, img_t.shape[1], img_t.shape[0])
                    # Skip invalid boxes
                    if bw <= 0 or bh <= 0:
                        continue
                    yolo_boxes.append((cls, xc, yc, bw, bh))

                aug_img_name = f"{Path(img_name).stem}_aug{k}{Path(img_name).suffix}"
                aug_lbl_name = Path(img_name).with_suffix('.txt').name.replace('.txt', f"_aug{k}.txt")
                cv2.imwrite(str(Path(out_images_dir, aug_img_name)), img_t)
                write_label_file(Path(out_labels_dir, aug_lbl_name), yolo_boxes)
            except Exception:
                # Skip failures
                continue


def split_train_val(in_images_dir, in_labels_dir, out_base_dir, train_ratio=0.8):
    images = [f for f in os.listdir(in_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    n_train = int(len(images) * train_ratio)
    train_imgs = images[:n_train]
    val_imgs = images[n_train:]

    for subset, subset_imgs in [('train', train_imgs), ('val', val_imgs)]:
        img_out = Path(out_base_dir, 'data', 'images', subset)
        lbl_out = Path(out_base_dir, 'data', 'labels', subset)
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img in subset_imgs:
            shutil.copy2(Path(in_images_dir, img), Path(img_out, img))
            lbl = Path(in_labels_dir, Path(img).with_suffix('.txt').name)
            if lbl.exists():
                shutil.copy2(lbl, Path(lbl_out, lbl.name))


def update_dataset_yaml(yaml_path, nc, names):
    cfg = {
        'train': 'data/images/train',
        'val': 'data/images/val',
        'nc': nc,
        'names': names,
    }
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description='Prepare dataset: augment and split for YOLO')
    ap.add_argument('--in-images', required=True, help='Source images dir')
    ap.add_argument('--in-labels', required=True, help='Source labels dir (YOLO format)')
    ap.add_argument('--out-images', default='prepared/images', help='Augmented images output dir')
    ap.add_argument('--out-labels', default='prepared/labels', help='Augmented labels output dir')
    ap.add_argument('--aug-per-image', type=int, default=1, help='Number of augmented samples per image')
    ap.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio')
    ap.add_argument('--update-yaml', action='store_true', help='Update dataset.yaml nc and names')
    ap.add_argument('--nc', type=int, default=5, help='Number of classes')
    ap.add_argument('--names', nargs='+', default=['Naruto','Goku','Luffy','Sukuna','Gojo'], help='Class names')
    args = ap.parse_args()

    Path(args.out_images).mkdir(parents=True, exist_ok=True)
    Path(args.out_labels).mkdir(parents=True, exist_ok=True)

    augment_dataset(args.in_images, args.in_labels, args.out_images, args.out_labels, args.aug_per_image)

    # Split to project data layout
    split_train_val(args.out_images, args.out_labels, out_base_dir='.', train_ratio=args.train_ratio)

    if args.update_yaml:
        update_dataset_yaml('dataset.yaml', args.nc, args.names)
    print('Dataset preparation complete.')


if __name__ == '__main__':
    main()
