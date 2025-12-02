import os
from pathlib import Path
from ultralytics import YOLO
import cv2

MODEL_PATH = os.path.join('runs','detect','train','weights','best.pt')
IN_DIR = os.path.join('data','raw','images','val')
OUT_DIR = os.path.join('runs','batch_preds')
os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

def main():
    for name in os.listdir(IN_DIR):
        if not name.lower().endswith(('.jpg','.jpeg','.png','.webp')):
            continue
        p = os.path.join(IN_DIR, name)
        results = model(p)
        # Save annotated image
        for r in results:
            im = r.plot()
            cv2.imwrite(os.path.join(OUT_DIR, name), im)
    print('Batch predictions saved to', OUT_DIR)

if __name__ == '__main__':
    main()
