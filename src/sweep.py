import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description='Quick sweep over model sizes')
    p.add_argument('--data', type=str, default='dataset.yaml', help='Dataset YAML')
    p.add_argument('--epochs', type=int, default=5, help='Epochs per model')
    p.add_argument('--imgsz', type=int, default=640, help='Image size')
    p.add_argument('--models', nargs='+', default=['yolov8n.pt','yolov8s.pt','yolov8m.pt'], help='Pretrained models to compare')
    return p.parse_args()


def main():
    args = parse_args()
    summary = []
    for m in args.models:
        model = YOLO(m)
        res = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, name=f'sweep_{m.split(".")[0]}')
        summary.append((m, res))
    print('Sweep complete. Check runs/detect for per-model results and choose best trade-off.')


if __name__ == '__main__':
    main()