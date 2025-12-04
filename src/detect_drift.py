"""
Data Drift Detection Script
- Compares embedding distributions between training and new incoming data
- Uses simple statistical tests (KS-test) and distance metrics
"""
import os
from pathlib import Path
import json
import numpy as np
import cv2
from scipy.stats import ks_2samp

TRAIN_DIR = 'data/raw/images/train'
NEW_DIR = 'data/raw/images/val'
OUT_PATH = 'runs/drift/drift_report.json'
SAMPLE_N = 500


def image_histogram(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        return np.zeros((256,), dtype=np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    return hist


def sample_paths(folder: str, n: int) -> list:
    all_paths = [str(p) for p in Path(folder).glob('*.jpg')]
    if len(all_paths) == 0:
        return []
    np.random.seed(42)
    np.random.shuffle(all_paths)
    return all_paths[: min(n, len(all_paths))]


def compute_distributions(folder: str, n: int):
    paths = sample_paths(folder, n)
    hists = [image_histogram(p) for p in paths]
    return np.stack(hists) if hists else np.zeros((0, 256))


def ks_tests(train_dists: np.ndarray, new_dists: np.ndarray):
    # KS-test per-bin aggregate (approximation)
    p_values = []
    for i in range(train_dists.shape[1]):
        p_values.append(ks_2samp(train_dists[:, i], new_dists[:, i]).pvalue)
    return float(np.mean(p_values))


def l2_distance(train_dists: np.ndarray, new_dists: np.ndarray):
    return float(np.linalg.norm(train_dists.mean(axis=0) - new_dists.mean(axis=0)))


def run():
    os.makedirs(Path(OUT_PATH).parent, exist_ok=True)
    train_dists = compute_distributions(TRAIN_DIR, SAMPLE_N)
    new_dists = compute_distributions(NEW_DIR, SAMPLE_N)
    if train_dists.shape[0] == 0 or new_dists.shape[0] == 0:
        print("Insufficient data to compute drift.")
        return
    ks_p_mean = ks_tests(train_dists, new_dists)
    l2 = l2_distance(train_dists, new_dists)
    report = {
        'ks_p_mean': ks_p_mean,
        'l2_mean_hist_distance': l2,
        'thresholds': {
            'ks_p_mean': 0.05,
            'l2_mean_hist_distance': 0.02,
        },
        'drift_detected': ks_p_mean < 0.05 or l2 > 0.02,
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print("Drift report written to", OUT_PATH)


if __name__ == '__main__':
    run()
