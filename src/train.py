import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on anime characters")
    p.add_argument('--model', type=str, default='yolov8n.pt', help='Pretrained model to fine-tune')
    p.add_argument('--data', type=str, default='dataset.yaml', help='Dataset YAML path')
    p.add_argument('--epochs', type=int, default=50, help='Training epochs')
    p.add_argument('--imgsz', type=int, default=640, help='Image size')
    p.add_argument('--batch', type=int, default=16, help='Batch size')
    p.add_argument('--workers', type=int, default=4, help='Dataloader workers')
    p.add_argument('--project', type=str, default='runs/detect', help='Project folder')
    p.add_argument('--name', type=str, default='train', help='Run name')
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        optimizer='auto',
        device=0 if YOLO(args.model).device.type != 'cpu' else 'cpu',
        plots=True,
        pretrained=True,
    )

    print("Training complete. Metrics:")
    print(results)


if __name__ == '__main__':
    main()
