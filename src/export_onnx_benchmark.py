"""
Export trained YOLOv8 model to ONNX and run performance benchmarks.
Measures load time, inference latency, and throughput on CPU.
"""
import os
import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2

WEIGHTS = 'runs/detect/train2/weights/best.pt'
ONNX_OUT = 'models/best.onnx'
SAMPLE_DIR = 'data/raw/images/val'
N_WARMUP = 5
N_RUNS = 50


def ensure_weights(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights not found: {path}. Train the model first.")


def export_to_onnx(weights: str, out_path: str):
    model = YOLO(weights)
    print(f"Exporting to ONNX: {out_path}")
    model.export(format='onnx', imgsz=640, dynamic=False, opset=12)
    # Ultralytics saves to same directory; move if needed
    # Try common output name
    candidate = Path(weights).with_suffix('.onnx')
    if candidate.exists():
        os.makedirs(Path(out_path).parent, exist_ok=True)
        candidate.replace(out_path)
    print("ONNX export completed.")


def load_images(sample_dir: str, limit: int = 50):
    paths = []
    for p in Path(sample_dir).glob('*.jpg'):
        paths.append(str(p))
        if len(paths) >= limit:
            break
    if not paths:
        raise RuntimeError(f"No sample images found at {sample_dir}")
    return paths


def benchmark(weights: str, sample_dir: str):
    print("Loading YOLO model...")
    t0 = time.time()
    model = YOLO(weights)
    load_time = time.time() - t0

    images = load_images(sample_dir, limit=50)
    latencies = []

    # Warmup
    for i in range(N_WARMUP):
        _ = model.predict(source=images[0], imgsz=640, verbose=False)

    print(f"Running {N_RUNS} inference runs...")
    for i in range(N_RUNS):
        img = images[i % len(images)]
        t1 = time.time()
        _ = model.predict(source=img, imgsz=640, verbose=False)
        latencies.append(time.time() - t1)

    avg = float(np.mean(latencies))
    p95 = float(np.percentile(latencies, 95))
    throughput = 1.0 / avg if avg > 0 else 0.0

    print("\nBenchmark Results")
    print(f"Model load time: {load_time:.3f}s")
    print(f"Average latency: {avg*1000:.2f} ms")
    print(f"p95 latency:     {p95*1000:.2f} ms")
    print(f"Throughput:      {throughput:.2f} images/sec")

    return {
        'load_time_s': load_time,
        'avg_latency_ms': avg * 1000.0,
        'p95_latency_ms': p95 * 1000.0,
        'throughput_img_per_s': throughput,
    }


if __name__ == '__main__':
    ensure_weights(WEIGHTS)
    export_to_onnx(WEIGHTS, ONNX_OUT)
    benchmark(WEIGHTS, SAMPLE_DIR)
