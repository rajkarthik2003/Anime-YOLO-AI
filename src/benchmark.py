import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Simple Level-1 benchmark: EDA + baseline classifier on manifest
CSV = os.path.join('data','raw','manifest_filtered.csv')
OUT_DIR = os.path.join('runs','benchmark')
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ["naruto","luffy","gojo","goku","sukuna"]

def has_target(tags: str) -> int:
    if not isinstance(tags, str):
        return 0
    t = tags.lower()
    return int(any(cls in t for cls in CLASSES))

def main():
    df = pd.read_csv(CSV)
    df['target'] = df['tags'].apply(has_target)
    desc = df.describe(include='all')
    desc.to_csv(os.path.join(OUT_DIR,'describe.csv'))

    df['file_name'] = df['file'].apply(lambda p: os.path.basename(str(p)))
    df['file_len'] = df['file_name'].str.len()
    df['tags_len'] = df['tags'].astype(str).str.len()
    df['domain'] = 'unknown'

    X = df[['file_len','tags_len','domain']]
    y = df['target']

    num_features = ['file_len','tags_len']
    cat_features = ['domain']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_features),
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_features)
        ]
    )

    model = LogisticRegression(max_iter=1000)
    pipe = Pipeline(steps=[('prep', preprocessor), ('clf', model)])

    # Ensure both classes present; if not, derive a heuristic label
    if y.nunique() < 2:
        # Use tags_len median to split into two classes for demo purposes
        median_len = df['tags_len'].median()
        y = (df['tags_len'] > median_len).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    with open(os.path.join(OUT_DIR,'metrics.txt'),'w') as f:
        f.write(f"accuracy={acc}\nprecision={pr}\nrecall={rc}\nf1={f1}\n")

    print(f"Benchmark complete. Accuracy={acc:.3f} F1={f1:.3f}")

if __name__ == '__main__':
    main()
import time
import argparse
import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description='Benchmark inference FPS on webcam/video')
    p.add_argument('--weights', type=str, default='runs/detect/train/weights/best.pt', help='Model weights path')
    p.add_argument('--source', type=str, default='0', help='0 for webcam or path to video')
    p.add_argument('--warmup', type=int, default=10, help='Warmup frames')
    p.add_argument('--measure', type=int, default=200, help='Measured frames')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            raise RuntimeError('Failed to open source')
        # Warmup
        for _ in range(args.warmup):
            ok, frame = cap.read()
            if not ok:
                break
            _ = model.predict(frame, conf=args.conf)
        # Measure
        start = time.time()
        frames = 0
        for _ in range(args.measure):
            ok, frame = cap.read()
            if not ok:
                break
            _ = model.predict(frame, conf=args.conf)
            frames += 1
        elapsed = time.time() - start
        fps = frames / elapsed if elapsed > 0 else 0
        print(f'Frames: {frames}, Time: {elapsed:.2f}s, FPS: {fps:.2f}')
        cap.release()
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            raise RuntimeError('Failed to open source')
        # Warmup
        for _ in range(args.warmup):
            ok, frame = cap.read()
            if not ok:
                break
            _ = model.predict(frame, conf=args.conf)
        # Measure
        start = time.time()
        frames = 0
        while frames < args.measure:
            ok, frame = cap.read()
            if not ok:
                break
            _ = model.predict(frame, conf=args.conf)
            frames += 1
        elapsed = time.time() - start
        fps = frames / elapsed if elapsed > 0 else 0
        print(f'Frames: {frames}, Time: {elapsed:.2f}s, FPS: {fps:.2f}')
        cap.release()


if __name__ == '__main__':
    main()