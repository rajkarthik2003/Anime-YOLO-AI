import os
import cv2
import pandas as pd

MANIFEST = os.path.join("data", "raw", "manifest_filtered.csv")
OUT_LABELS_DIR = os.path.join("data", "raw", "labels")

os.makedirs(OUT_LABELS_DIR, exist_ok=True)

# Fallback to Haar cascade (simple, fast)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

df = pd.read_csv(MANIFEST)

for _, r in df.iterrows():
    img_path = r['file']
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(24,24))
    bboxes = []
    for (x,y,ww,hh) in rects:
        x_c = (x + ww/2)/w
        y_c = (y + hh/2)/h
        w_n = ww / w
        h_n = hh / h
        bboxes.append((0, x_c, y_c, w_n, h_n))
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_txt = os.path.join(OUT_LABELS_DIR, base + ".txt")
    with open(out_txt, "w") as f:
        for bbox in bboxes:
            f.write(" ".join(str(v) for v in bbox) + "\n")

print("Auto-labeling complete (check data/raw/labels).")
