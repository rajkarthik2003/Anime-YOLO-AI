from ultralytics import YOLO
import cv2
import argparse

# Global model instance for API import
model = YOLO('runs/detect/train2/weights/best.pt')

def parse_args():
    p = argparse.ArgumentParser(description="Real-time inference for anime detection")
    p.add_argument('--weights', type=str, default='runs/detect/train2/weights/best.pt', help='Path to trained weights')
    p.add_argument('--source', type=str, default='0', help='Source: 0 for webcam, path for image/video')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    p.add_argument('--show', action='store_true', help='Show window output')
    return p.parse_args()


def open_source(src_str: str):
    if src_str.isdigit():
        return cv2.VideoCapture(int(src_str))
    else:
        return src_str  # image/video path for yolo.predict


def main():
    args = parse_args()
    model = YOLO(args.weights)

    # If source is a camera or video, use cv2 loop; else, run single predict
    if args.source.isdigit():
        cap = open_source(args.source)
        if not cap or not cap.isOpened():
            raise RuntimeError("Failed to open video source")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = model.predict(frame, conf=args.conf)
            annotated = results[0].plot()
            if args.show:
                cv2.imshow("Anime YOLO Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
    else:
        results = model.predict(args.source, conf=args.conf, stream=False)
        for r in results:
            annotated = r.plot()
            if args.show:
                cv2.imshow("Anime YOLO Detection", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# Initialize global model for API
model = YOLO('runs/detect/train2/weights/best.pt')
