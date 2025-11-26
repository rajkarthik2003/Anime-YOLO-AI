from ultralytics import YOLO
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Multi-object tracking with YOLOv8 + ByteTrack")
    p.add_argument('--weights', type=str, default='runs/detect/train/weights/best.pt', help='Path to trained weights')
    p.add_argument('--source', type=str, default='0', help='0 for webcam or path to video')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    p.add_argument('--show', action='store_true', help='Visualize tracking window')
    p.add_argument('--save', action='store_true', help='Save output video to runs/track')
    p.add_argument('--tracker', type=str, default='bytetrack.yaml', help='Tracker config (e.g., bytetrack.yaml)')
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.track(
        source=args.source,
        conf=args.conf,
        tracker=args.tracker,
        show=args.show,
        save=args.save,
        persist=True,
    )


if __name__ == '__main__':
    main()