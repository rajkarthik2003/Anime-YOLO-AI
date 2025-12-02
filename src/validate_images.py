import os
import cv2

IMG_DIR = os.path.join("data", "raw", "images")

def is_valid_image(path: str) -> bool:
    # non-empty and readable by OpenCV
    try:
        if not os.path.isfile(path):
            return False
        if os.path.getsize(path) == 0:
            return False
        # Skip GIFs (frequently unsupported/animated)
        if path.lower().endswith(('.gif')):
            return False
        img = cv2.imread(path)
        if img is None:
            return False
        return True
    except Exception:
        return False


def main():
    removed = 0
    for root, _, files in os.walk(IMG_DIR):
        for name in files:
            ext = name.lower().split(".")[-1]
            if ext not in {"jpg", "jpeg", "png", "bmp", "webp", "gif"}:
                continue
            full = os.path.join(root, name)
            if not is_valid_image(full):
                try:
                    os.remove(full)
                    removed += 1
                except Exception:
                    pass
    print(f"Removed {removed} invalid images.")


if __name__ == "__main__":
    main()
