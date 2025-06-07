# для поиска чернобелых изображений (для датасетов)
import os
import cv2
import shutil
from PIL import Image
import numpy as np
def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
def is_grayscale(image_path, tolerance=10):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        if len(img.shape) < 3 or img.shape[2] == 1:
            return True
        b, g, r = cv2.split(img)
        diff_rg = np.abs(r.astype(int) - g.astype(int))
        diff_rb = np.abs(r.astype(int) - b.astype(int))
        diff_gb = np.abs(g.astype(int) - b.astype(int))
        # Допускаем небольшое различие в каналах
        if (diff_rg < tolerance).all() and (diff_rb < tolerance).all() and (diff_gb < tolerance).all():
            return True
        return False
    except Exception as e:
        print(f"Skipped corrupted file: {image_path} — {e}")
        return False
def move_grayscale_images(source_folder):
    no_color_folder = source_folder.rstrip("\\/") + "(no color)"
    os.makedirs(no_color_folder, exist_ok=True)
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            if is_valid_image(file_path) and is_grayscale(file_path):
                destination = os.path.join(no_color_folder, filename)
                shutil.move(file_path, destination)
                print(f"Moved grayscale image: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
source_path = r"/styles(3)/Romanticism"
move_grayscale_images(source_path)
