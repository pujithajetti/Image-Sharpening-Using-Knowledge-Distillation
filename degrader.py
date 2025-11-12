# degrader.py
import os, cv2, numpy as np, random
from pathlib import Path
from PIL import Image, ImageEnhance

def add_noise(img):
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def add_motion_blur(img, k=5):
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = np.ones(k)
    return cv2.filter2D(img, -1, kernel / k)

def degrade_image(img):
    if random.random() < 0.9:
        img = add_noise(img)
    if random.random() < 0.8:
        img = add_motion_blur(img, random.choice([3, 5, 7]))
    return img

def generate_blurry_dataset():
    src = Path("datasets/original")
    dst = Path("datasets/noisy")
    dst.mkdir(parents=True, exist_ok=True)
    images = list(src.glob("*.jpg")) + list(src.glob("*.png"))

    for i, path in enumerate(images):
        img = cv2.imread(str(path))
        degraded = degrade_image(img)
        cv2.imwrite(str(dst / path.name), degraded)
        print(f"[{i+1}/{len(images)}] Saved: {dst / path.name}")

if __name__ == "__main__":
    generate_blurry_dataset()
