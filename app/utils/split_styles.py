# подготовка датасетов к обучению  20/80
import os
import shutil
import random
from tqdm import tqdm
SOURCE_DIR = '../../zip_styles' # путь(папка)/ что необходимо поделить 20/80
TARGET_DIR = '../../dataset_styles' # путь(папка)/ куда необходимо записать
SPLIT_RATIO = 0.8  # 80% train, 20% val
def clear_previous_dataset():
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(os.path.join(TARGET_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'val'), exist_ok=True)
def prepare_dataset():
    clear_previous_dataset()
    print("Копирование изображений...")
    for class_name in tqdm(os.listdir(SOURCE_DIR), desc="Стили"):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        split_point = int(len(images) * SPLIT_RATIO)
        train_images = images[:split_point]
        val_images = images[split_point:]

        for phase, imgs in zip(['train', 'val'], [train_images, val_images]):
            phase_dir = os.path.join(TARGET_DIR, phase, class_name)
            os.makedirs(phase_dir, exist_ok=True)
            for img in tqdm(imgs, desc=f"→ {phase}/{class_name}", leave=False):
                shutil.copy2(os.path.join(class_path, img), os.path.join(phase_dir, img))
    print("Style dataset_genre split complete.\n")
def count_images(dataset_path):
    total = 0
    per_class = {}
    for root, _, files in os.walk(dataset_path):
        images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            class_name = os.path.basename(root)
            count = len(images)
            per_class[class_name] = count
            total += count
    return total, per_class
if __name__ == "__main__":
    prepare_dataset()
    for phase in ['train', 'val']:
        path = os.path.join(TARGET_DIR, phase)
        total_images, class_counts = count_images(path)
        print(f" Изображений в {phase}: {total_images}")
        for cls, count in sorted(class_counts.items()):
            print(f"{cls}: {count}")
        print()
