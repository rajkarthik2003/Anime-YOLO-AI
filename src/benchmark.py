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