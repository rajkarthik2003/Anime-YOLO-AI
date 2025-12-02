import os, glob
import cv2
import albumentations as A

IN_IMG = os.path.join("data", "raw", "images", "train")
IN_LBL = os.path.join("data", "raw", "labels", "train")
OUT_IMG = os.path.join("data", "prepared", "images")
OUT_LBL = os.path.join("data", "prepared", "labels")
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

transform = A.Compose([
    A.RandomBrightnessContrast(),
    A.MotionBlur(p=0.2),
    A.GaussNoise(p=0.2),
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=0.1, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

imgs = glob.glob(os.path.join(IN_IMG, "*.jpg")) + glob.glob(os.path.join(IN_IMG, "*.png"))
cnt = 0
for img_path in imgs:
    base = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(IN_LBL, base + ".txt")
    img = cv2.imread(img_path)
    if img is None:
        continue
    bboxes = []
    class_labels = []
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x, y, bw, bh = parts
                    bboxes.append([float(x), float(y), float(bw), float(bh)])
                    class_labels.append(int(cls))
    n_augs = 2
    for i in range(n_augs):
        try:
            augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            out_img = augmented['image']
            out_boxes = augmented['bboxes']
            out_labels = augmented['class_labels']
            out_name = f"{base}_aug{cnt}.jpg"
            cv2.imwrite(os.path.join(OUT_IMG, out_name), out_img)
            with open(os.path.join(OUT_LBL, out_name.replace('.jpg','.txt')), "w") as f:
                for lab, bbox in zip(out_labels, out_boxes):
                    f.write(f"{lab} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            cnt += 1
        except Exception:
            continue

print("Augmentation done.")
