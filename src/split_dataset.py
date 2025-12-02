import os, random, shutil, glob

IMG_DIR = os.path.join("data", "raw", "images")
LBL_DIR = os.path.join("data", "raw", "labels")
OUT_IMG_TRAIN = os.path.join("data", "raw", "images", "train")
OUT_IMG_VAL = os.path.join("data", "raw", "images", "val")
OUT_LBL_TRAIN = os.path.join("data", "raw", "labels", "train")
OUT_LBL_VAL = os.path.join("data", "raw", "labels", "val")

for d in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_LBL_TRAIN, OUT_LBL_VAL]:
    os.makedirs(d, exist_ok=True)

images = glob.glob(os.path.join(IMG_DIR, "*.jpg")) + glob.glob(os.path.join(IMG_DIR, "*.png"))
random.seed(42)
random.shuffle(images)
split = int(0.8 * len(images))
train_imgs = images[:split]
val_imgs = images[split:]

def move_list(imgs, dest_img_dir, dest_lbl_dir):
    for img in imgs:
        base = os.path.splitext(os.path.basename(img))[0]
        lbl = os.path.join(LBL_DIR, base + ".txt")
        shutil.copy(img, os.path.join(dest_img_dir, os.path.basename(img)))
        if os.path.exists(lbl):
            shutil.copy(lbl, os.path.join(dest_lbl_dir, os.path.basename(lbl)))
        else:
            open(os.path.join(dest_lbl_dir, base + ".txt"), "w").close()

move_list(train_imgs, OUT_IMG_TRAIN, OUT_LBL_TRAIN)
move_list(val_imgs, OUT_IMG_VAL, OUT_LBL_VAL)
print("Split completed.")
